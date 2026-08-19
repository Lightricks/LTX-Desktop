import { app } from 'electron'
import fs from 'fs'
import path from 'path'

export interface AppState {
  analyticsEnabled?: boolean
  installationId?: string
  projectAssetsPath?: string
  skippedUpdateVersion?: string
  autoCheckUpdates?: boolean
  [key: string]: unknown
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

export function getSkippedUpdateVersion(): string | undefined {
  return readAppState().skippedUpdateVersion
}

export function setSkippedUpdateVersion(version: string | undefined): void {
  const s = readAppState()
  if (version) s.skippedUpdateVersion = version
  else delete s.skippedUpdateVersion
  writeAppState(s)
}

export function getAutoCheckUpdates(): boolean {
  // Default ON: existing installs and fresh installs behave as before.
  return readAppState().autoCheckUpdates ?? true
}

export function setAutoCheckUpdates(enabled: boolean): void {
  const s = readAppState()
  s.autoCheckUpdates = enabled
  writeAppState(s)
}
