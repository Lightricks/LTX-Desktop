import { autoUpdater, UpdateInfo, ProgressInfo } from 'electron-updater'
import { app } from 'electron'
import { logger } from './logger'
import { preDownloadPythonForUpdate } from './python-setup'
import { getMainWindow } from './window'
import {
  getSkippedUpdateVersion, setSkippedUpdateVersion,
  getAutoCheckUpdates, setAutoCheckUpdates,
} from './app-state'
import { isGenerationActive } from './python-backend'
import type { UpdateStatePayload } from '../shared/electron-api-schema'

export type UpdateChannel = 'latest' | 'beta' | 'alpha'

// Cap untrusted feed notes so a huge GitHub body cannot bloat IPC / the modal.
const MAX_RELEASE_NOTES_CHARS = 16_384

// The single in-memory value of update state. Broadcast on every change.
let state: UpdateStatePayload = { status: 'idle', currentVersion: app.getVersion() }
let periodicHandle: ReturnType<typeof setInterval> | null = null

function setState(patch: Partial<UpdateStatePayload>): void {
  state = { ...state, ...patch }
  getMainWindow()?.webContents.send('update-event', state)
}

function releaseNotesFromFeed(info: UpdateInfo): string | undefined {
  if (typeof info.releaseNotes !== 'string' || info.releaseNotes.length === 0) return undefined
  if (info.releaseNotes.length <= MAX_RELEASE_NOTES_CHARS) return info.releaseNotes
  return `${info.releaseNotes.slice(0, MAX_RELEASE_NOTES_CHARS)}\n…`
}

export function getUpdateState(): UpdateStatePayload {
  return state
}

// True when we must not start or clobber with a new check.
function isBusy(): boolean {
  return state.status === 'checking' || state.status === 'downloading' || state.status === 'downloaded'
}

function hasRestorableOffer(): boolean {
  return Boolean(state.version && getSkippedUpdateVersion() !== state.version)
}

// Network/feed errors must not drop a known offer or a finished download.
function failUpdate(message: string): void {
  logger.error(`[updater] ${message}`)
  if (state.status === 'downloading') {
    setState({ status: 'available', message })
    return
  }
  if (state.status === 'downloaded') {
    setState({ message })
    return
  }
  if (hasRestorableOffer()) {
    setState({ status: 'available', message })
    return
  }
  setState({ status: 'idle', message })
}

function runCheck(): void {
  if (isBusy()) return // never interrupt an in-flight download or a downloaded-and-waiting state
  logger.info('[updater] Checking for update...')
  autoUpdater.checkForUpdates().catch((e) => {
    failUpdate(e instanceof Error ? e.message : String(e))
  })
}

// Start/stop the 4h timer to match the autoCheckUpdates setting. Safe to call repeatedly.
export function armPeriodicCheck(): void {
  if (periodicHandle) { clearInterval(periodicHandle); periodicHandle = null }
  if (getAutoCheckUpdates()) {
    periodicHandle = setInterval(runCheck, 4 * 60 * 60 * 1000)
  }
}

export function initAutoUpdater(channel: UpdateChannel = 'latest'): void {
  if (channel !== 'latest') {
    autoUpdater.channel = channel
    autoUpdater.allowPrerelease = true
  }

  // Core change: the user controls download and install.
  autoUpdater.autoDownload = false
  autoUpdater.autoInstallOnAppQuit = false

  autoUpdater.on('checking-for-update', () => setState({ status: 'checking', message: undefined }))

  autoUpdater.on('update-available', (info: UpdateInfo) => {
    const version = info.version
    // Skip enforcement: if the user skipped THIS version, stay silent (idle).
    if (getSkippedUpdateVersion() === version) {
      setState({ status: 'idle', version })
      return
    }
    setState({
      status: 'available',
      version,
      // Release notes are untrusted feed content — accept plain strings only, render as text.
      releaseNotes: releaseNotesFromFeed(info),
      message: undefined,
    })
  })

  autoUpdater.on('update-not-available', () => setState({ status: 'not-available', message: undefined }))

  autoUpdater.on('download-progress', (p: ProgressInfo) =>
    setState({ status: 'downloading', percent: Math.round(p.percent) }),
  )

  autoUpdater.on('update-downloaded', async (info: UpdateInfo) => {
    setState({ status: 'downloaded', version: info.version })
    if (process.platform === 'darwin') return // macOS: no python pre-download (unchanged)

    logger.info(`[updater] Update downloaded: v${info.version}, pre-downloading python deps...`)
    try {
      const didDownload = await preDownloadPythonForUpdate(info.version, (progress) => {
        getMainWindow()?.webContents.send('python-update-progress', progress)
      })
      logger.info(didDownload
        ? '[updater] Python pre-download complete'
        : '[updater] No python changes needed')
    } catch (err) {
      logger.error(`[updater] Python pre-download failed: ${err}`)
    }
  })

  autoUpdater.on('error', (err: Error) => {
    failUpdate(err?.message ?? 'Update failed')
  })

  armPeriodicCheck()
  // One check shortly after startup, but only if auto-check is on.
  setTimeout(() => { if (getAutoCheckUpdates()) runCheck() }, 5_000)
}

// ---- Actions called from IPC handlers ----

// Manual "Check for updates": explicit user intent. Clear any skip and force a check even if
// auto-check is off.
export async function checkForUpdatesNow(): Promise<void> {
  if (isBusy()) return
  setSkippedUpdateVersion(undefined) // an explicit check overrides a previous skip
  logger.info('[updater] Manual check for updates...')
  try {
    await autoUpdater.checkForUpdates()
  } catch (e) {
    failUpdate(e instanceof Error ? e.message : String(e))
    throw e
  }
}

export async function startUpdateDownload(): Promise<void> {
  if (state.status !== 'available') return
  setState({ status: 'downloading', percent: 0, message: undefined })
  await autoUpdater.downloadUpdate()
}

// Guarded in MAIN: never quit mid-generation. Returns a result the renderer can surface.
export function installUpdateAndRestart(): { success: true } | { success: false; error: string } {
  if (state.status !== 'downloaded') {
    return { success: false, error: 'No update is ready to install.' }
  }
  if (isGenerationActive()) {
    return { success: false, error: 'A generation is running. Please wait for it to finish.' }
  }
  try {
    autoUpdater.quitAndInstall() // quits the app; does not return on success
    return { success: true }
  } catch (e) {
    const error = e instanceof Error ? e.message : String(e)
    logger.error(`[updater] install failed: ${error}`)
    return { success: false, error }
  }
}

export function skipUpdateVersion(version: string): void {
  setSkippedUpdateVersion(version)
  setState({ status: 'idle', version })
}

// Persist the auto-check setting AND start/stop the timer immediately (no restart needed).
export function setAutoCheckUpdatesEnabled(enabled: boolean): void {
  setAutoCheckUpdates(enabled)
  armPeriodicCheck()
}
