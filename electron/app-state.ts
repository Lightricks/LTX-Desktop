import { app } from 'electron'
import fs from 'fs'
import path from 'path'
import { getAppDataDir } from './app-paths'

export interface AppState {
  analyticsEnabled?: boolean
  installationId?: string
  projectAssetsPath?: string
  /** Custom folder of the asset store. Unset means `<userData>/assets`. */
  assetsDir?: string
  skippedUpdateVersion?: string
  autoCheckUpdates?: boolean
  setupComplete?: boolean
  /** Mirrors the in-app theme preference so Mac glass can follow it on launch. */
  windowTheme?: 'light' | 'dark' | 'system'
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
  // Write a temp file and rename it. A crash must not leave a half file: an unreadable state
  // reads as empty, and that would drop the saved assets folder.
  const statePath = getAppStatePath()
  const tempPath = `${statePath}.tmp`
  fs.writeFileSync(tempPath, JSON.stringify(state, null, 2))
  fs.renameSync(tempPath, statePath)
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

export function getDefaultAssetsDir(): string {
  return path.join(getAppDataDir(), 'assets')
}

let cachedAssetsDir: string | null = null

/** Folder of the asset store. A custom folder holds the `.ltx-assets` marker. */
export function getAssetsDir(): string {
  if (cachedAssetsDir) return cachedAssetsDir
  const saved = readAppState().assetsDir
  cachedAssetsDir = saved ? path.resolve(saved) : getDefaultAssetsDir()
  return cachedAssetsDir
}

export function isCustomAssetsDir(): boolean {
  return getAssetsDir() !== getDefaultAssetsDir()
}

/** `undefined` returns to the default folder. */
export function setAssetsDir(dir: string | undefined): void {
  const state = readAppState()
  if (dir && path.resolve(dir) !== getDefaultAssetsDir()) state.assetsDir = path.resolve(dir)
  else delete state.assetsDir
  writeAppState(state)
  cachedAssetsDir = null
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

export function getWindowTheme(): 'light' | 'dark' | 'system' | undefined {
  const theme = readAppState().windowTheme
  if (theme === 'light' || theme === 'dark' || theme === 'system') return theme
  return undefined
}

export function setWindowTheme(theme: 'light' | 'dark' | 'system'): void {
  const s = readAppState()
  s.windowTheme = theme
  writeAppState(s)
}
