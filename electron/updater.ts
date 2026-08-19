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
// Mac toggle-off sets UI to idle without aborting HTTP. Track the transfer so a
// new check cannot start while a zip is still downloading.
let macInFlight = false

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

function beginMacFlight(): void {
  if (process.platform === 'darwin') macInFlight = true
}

function endMacFlight(): void {
  macInFlight = false
}

// True when we must not start or clobber with a new check.
function isBusy(): boolean {
  return macInFlight
    || state.status === 'checking'
    || state.status === 'downloading'
    || state.status === 'downloaded'
}

function hasRestorableOffer(): boolean {
  return Boolean(state.version && getSkippedUpdateVersion() !== state.version)
}

// Network/feed errors must not drop a known offer or a finished download.
function failUpdate(message: string): void {
  logger.error(`[updater] ${message}`)
  if (process.platform === 'darwin') {
    // No modal / Try again. Keep a finished download; otherwise idle so Check retries.
    if (state.status === 'downloaded') {
      setState({ message })
      return
    }
    endMacFlight()
    setState({ status: 'idle', message })
    return
  }
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
  beginMacFlight()
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

// Mac has no update modal: the About toggle is the whole decision.
// On = check, auto-download, feed Squirrel so quit installs (1.2.0-style).
// Off = none of that. Windows/Linux keep user-gated download/install.
// Flipping off cannot abort an in-flight HTTP download (no CancellationToken).
// Squirrel is only fed when the zip finishes with autoInstallOnAppQuit still true;
// after that, toggling off does not un-stage the update.
function syncMacSilentUpdates(): void {
  const enabled = process.platform === 'darwin' && getAutoCheckUpdates()
  autoUpdater.autoDownload = enabled
  autoUpdater.autoInstallOnAppQuit = enabled
}

export function initAutoUpdater(channel: UpdateChannel = 'latest'): void {
  if (channel !== 'latest') {
    autoUpdater.channel = channel
    autoUpdater.allowPrerelease = true
  }

  // Windows/Linux: user controls download and install. Mac: syncMacSilentUpdates.
  autoUpdater.autoDownload = false
  autoUpdater.autoInstallOnAppQuit = false
  syncMacSilentUpdates()

  autoUpdater.on('checking-for-update', () => {
    if (process.platform === 'darwin' && !getAutoCheckUpdates()) return
    setState({ status: 'checking', message: undefined })
  })

  autoUpdater.on('update-available', (info: UpdateInfo) => {
    const version = info.version
    if (process.platform === 'darwin') {
      // Skip does not apply. If the toggle is off, do not sit in Windows `available`
      // (that would disable Check and claim a download). autoDownload starts next.
      if (!getAutoCheckUpdates()) {
        endMacFlight()
        setState({ status: 'idle', version })
        return
      }
      setState({
        status: 'downloading',
        version,
        percent: 0,
        releaseNotes: releaseNotesFromFeed(info),
        message: undefined,
      })
      return
    }
    // Skip is a Windows/Linux modal action.
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

  autoUpdater.on('update-not-available', () => {
    endMacFlight()
    setState({ status: 'not-available', message: undefined })
  })

  autoUpdater.on('download-progress', (p: ProgressInfo) => {
    if (process.platform === 'darwin' && !getAutoCheckUpdates()) return
    setState({ status: 'downloading', percent: Math.round(p.percent) })
  })

  autoUpdater.on('update-downloaded', async (info: UpdateInfo) => {
    endMacFlight()
    if (process.platform === 'darwin') {
      // Squirrel is fed only when autoInstallOnAppQuit is still true at zip-complete.
      if (!getAutoCheckUpdates()) {
        setState({ status: 'idle', version: info.version })
        return
      }
      setState({ status: 'downloaded', version: info.version })
      return // macOS: no python pre-download (unchanged)
    }
    setState({ status: 'downloaded', version: info.version })

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

// Manual "Check for updates". Windows: allowed even if auto-check is off (modal still
// gates download). Mac: the toggle is the opt-out, so a check with it off is a no-op.
export async function checkForUpdatesNow(): Promise<void> {
  if (isBusy()) return
  if (process.platform === 'darwin' && !getAutoCheckUpdates()) return
  setSkippedUpdateVersion(undefined) // an explicit check overrides a previous skip
  beginMacFlight()
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
  syncMacSilentUpdates()
  if (process.platform !== 'darwin') return
  if (enabled) {
    runCheck() // no-op while macInFlight (HTTP still running after toggle-off)
    return
  }
  // Drop in-flight check/download UI. Leave `downloaded` — Squirrel is already staged.
  if (state.status === 'checking' || state.status === 'available' || state.status === 'downloading') {
    setState({ status: 'idle' })
  }
}
