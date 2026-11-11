import os from 'os'
import type { AssetsMoveState } from '../shared/electron-api-schema'
import { formatBytes } from '../shared/format-bytes'
import { getAppDataDir } from './app-paths'
import { getAssetsDir, getDefaultAssetsDir, isCustomAssetsDir, setAssetsDir } from './app-state'
import { runMove, type MoveDeps, type MoveStatePatch } from './assets-move-run'
import { busyReason, systemFolders } from './assets-move-guards'
import { checkCustomAssetsDir, preflightMove, type MovePlan } from './assets-relocation'
import { getCurrentDir } from './config'
import { freeDiskBytes } from './free-disk-space'
import { logger } from './logger'
import { pickFolder, showMessage } from './native-dialogs'
import {
  holdBackendStart,
  isBackendAdopted,
  startPythonBackend,
  stopPythonBackendAndWait,
} from './python-backend'
import { getMainWindow } from './window'

export type ChangeResult = { success: true; path: string } | { success: false; error: string }

const CANCELLED: ChangeResult = { success: false, error: 'cancelled' }

let state: AssetsMoveState | null = null
let cancelRequested = false
let releaseBackendHold = () => {}
// Set at once, before the first dialog opens, so two calls cannot start two moves.
let changing = false

/** The move state while a move runs. `null` when idle. A finished move is not kept. */
export function getAssetsMoveState(): AssetsMoveState | null {
  return state && state.phase !== 'done' ? state : null
}

export function cancelAssetsMove(): void {
  if (state?.canCancel) cancelRequested = true
}

function emit(patch: MoveStatePatch): void {
  state = { filesDone: 0, filesTotal: 0, bytesDone: 0, bytesTotal: 0, canCancel: true, ...state, ...patch }
  getMainWindow()?.webContents.send('assets-move-progress', state)
}

/** The real backend and settings. `runMove` takes these, so tests can give fakes. */
const moveDeps: MoveDeps = {
  stopBackend: async () => {
    releaseBackendHold = holdBackendStart()
    await stopPythonBackendAndWait()
  },
  startBackend: async () => {
    // Release the hold first: a start that waits (for example after a renderer reload) joins this one.
    releaseBackendHold()
    await startPythonBackend({ askUser: false })
  },
  saveAssetsDir: setAssetsDir,
  emit,
  isCancelled: () => cancelRequested,
  logError: (message) => logger.error(message),
}

async function reject(message: string): Promise<ChangeResult> {
  await showMessage({ type: 'info', message: 'The assets folder was not changed.', detail: message, buttons: ['OK'], noLink: true })
  return { success: false, error: message }
}

async function confirmMove(plan: MovePlan): Promise<boolean> {
  const count = plan.files.length
  const detail =
    `From: ${plan.from}\nTo: ${plan.to}\n\n` +
    (count === 0
      ? 'There are no assets to move.'
      : `${count} files (${formatBytes(plan.totalBytes)}) will be copied, checked, and then deleted from the old folder. ` +
        `${formatBytes(plan.freeBytes)} is free in the new folder.`) +
    '\n\nThe app server restarts. Do not close the app while the move runs.'
  const response = await showMessage({
    type: 'question',
    message: 'Move your assets to this folder?',
    detail,
    buttons: ['Move assets', 'Cancel'],
    defaultId: 1,
    cancelId: 1,
    noLink: true,
  })
  return response === 0
}

/** `'pick'` asks the user for a folder. `'default'` goes back to the default folder. */
export async function changeAssetsDir(target: 'pick' | 'default'): Promise<ChangeResult> {
  if (changing) return reject('An assets move is already running.')
  changing = true
  try {
    return await changeAssetsDirLocked(target)
  } catch (error) {
    // The renderer gets a fixed text. The detail stays in the log.
    logger.error(`Assets folder change failed: ${error}`)
    return reject('Something went wrong. See the logs for details.')
  } finally {
    releaseBackendHold()
    cancelRequested = false
    changing = false
  }
}

async function changeAssetsDirLocked(target: 'pick' | 'default'): Promise<ChangeResult> {
  const toCustom = target === 'pick'
  if (!toCustom && !isCustomAssetsDir()) return reject('The app already uses the default folder.')
  if (isBackendAdopted()) return reject('The app server was started outside this app, so the app cannot restart it.')

  const busy = await busyReason()
  if (busy) return reject(`${busy} Wait until it finishes, then try again.`)

  const to = toCustom ? await pickFolder('Choose the assets folder', { canCreate: true }) : getDefaultAssetsDir()
  if (!to) return CANCELLED

  const from = getAssetsDir()
  const previousDir = isCustomAssetsDir() ? from : undefined
  // The default folder is inside the app data folder, so only a picked folder is checked against it.
  const blockedRoots = toCustom ? systemFolders() : []
  const appFolders = toCustom ? [getAppDataDir(), getCurrentDir()] : []
  const replan = async () => {
    if (previousDir) {
      // A drive that was unplugged must not look like an empty library.
      const source = await checkCustomAssetsDir(previousDir)
      if (!source.ok) return { ok: false as const, reason: `The current assets folder is not available. ${source.reason}` }
    }
    return preflightMove({ from, to, protectedDirs: [os.homedir(), getAppDataDir(), getCurrentDir()], blockedRoots, appFolders, freeBytes: freeDiskBytes })
  }
  const preflight = await replan()
  if (!preflight.ok) return reject(preflight.reason)
  if (!(await confirmMove(preflight.plan))) return CANCELLED

  // The check above may be old after the user read the dialog.
  const busyAfterConfirm = await busyReason()
  if (busyAfterConfirm) return reject(`${busyAfterConfirm} Wait until it finishes, then try again.`)

  cancelRequested = false
  state = null
  emit({ phase: 'checking' })
  const result = await runMove({ plan: preflight.plan, toCustom, previousDir, replan }, moveDeps)
  if (result === 'moved') return { success: true, path: getAssetsDir() }
  return result === 'cancelled' ? CANCELLED : { success: false, error: state?.result?.message ?? 'The move failed.' }
}
