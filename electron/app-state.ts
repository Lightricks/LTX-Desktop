import { app, safeStorage } from 'electron'
import fs from 'fs'
import path from 'path'

export interface AppState {
  analyticsEnabled?: boolean
  installationId?: string
  projectAssetsPath?: string
  backendConnection?: StoredBackendConnection
  [key: string]: unknown
}

export type BackendConnectionMode = 'managed-local' | 'external'

interface StoredBackendConnection {
  mode: BackendConnectionMode
  url?: string
  authToken?: string
}

export type BackendConnectionConfig =
  | { mode: 'managed-local' }
  | { mode: 'external'; url: string; authToken: string }

export interface BackendConnectionSummary {
  mode: BackendConnectionMode
  url: string
  hasAuthToken: boolean
}

export function getAppStatePath(): string {
  return path.join(app.getPath('userData'), 'app_state.json')
}

export function readAppState(): AppState {
  const statePath = getAppStatePath()
  try {
    if (fs.existsSync(statePath)) {
      return JSON.parse(fs.readFileSync(statePath, 'utf-8')) as AppState
    }
  } catch (err) {
    console.warn('[app-state] failed to read app state:', err)
  }
  return {}
}

export function writeAppState(state: AppState): void {
  fs.writeFileSync(getAppStatePath(), JSON.stringify(state, null, 2))
}

function encryptSecret(value: string): string {
  if (!safeStorage.isEncryptionAvailable()) {
    throw new Error('Secure credential storage is unavailable on this system')
  }
  return safeStorage.encryptString(value).toString('base64')
}

function decryptSecret(value: string | undefined): string {
  if (!value) return ''
  if (!safeStorage.isEncryptionAvailable()) {
    throw new Error('Secure credential storage is unavailable on this system')
  }
  return safeStorage.decryptString(Buffer.from(value, 'base64'))
}

export function readBackendConnectionConfig(): BackendConnectionConfig {
  const stored = readAppState().backendConnection
  if (!stored || stored.mode !== 'external') {
    return { mode: 'managed-local' }
  }

  if (!stored.url || !stored.authToken) {
    throw new Error('The saved external backend connection is incomplete')
  }

  return {
    mode: 'external',
    url: stored.url,
    authToken: decryptSecret(stored.authToken),
  }
}

export function getBackendConnectionSummary(): BackendConnectionSummary {
  const stored = readAppState().backendConnection
  if (!stored || stored.mode !== 'external') {
    return {
      mode: 'managed-local',
      url: '',
      hasAuthToken: false,
    }
  }

  return {
    mode: 'external',
    url: stored.url ?? '',
    hasAuthToken: Boolean(stored.authToken),
  }
}

export function writeBackendConnectionConfig(config: BackendConnectionConfig): void {
  const state = readAppState()
  if (config.mode === 'managed-local') {
    state.backendConnection = { mode: 'managed-local' }
  } else {
    state.backendConnection = {
      mode: 'external',
      url: config.url,
      authToken: encryptSecret(config.authToken),
    }
  }
  writeAppState(state)
}

let cachedProjectAssetsPath: string | null = null

export function getProjectAssetsPath(): string {
  if (cachedProjectAssetsPath) return cachedProjectAssetsPath
  const state = readAppState()
  if (state.projectAssetsPath) {
    cachedProjectAssetsPath = path.resolve(state.projectAssetsPath)
    return cachedProjectAssetsPath
  }
  const defaultPath = path.resolve(path.join(app.getPath('downloads'), 'Ltx Desktop Assets'))
  cachedProjectAssetsPath = defaultPath
  return defaultPath
}

export function setProjectAssetsPath(p: string): void {
  const resolvedPath = path.resolve(p)
  cachedProjectAssetsPath = resolvedPath
  const state = readAppState()
  state.projectAssetsPath = resolvedPath
  writeAppState(state)
}
