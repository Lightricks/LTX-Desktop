import type { AssetsMoveState } from '../shared/electron-api-schema.ts'
import {
  copyFiles,
  deleteMovedFiles,
  releaseFolder,
  removeCreated,
  removeMarker,
  writeMarker,
  type MovePlan,
  type PreflightResult,
} from './assets-relocation.ts'

export type MoveStatePatch = Partial<AssetsMoveState> & Pick<AssetsMoveState, 'phase'>

/** What the move needs from the app. Tests give fakes, the app gives the real backend and settings. */
export type MoveDeps = {
  /** Stops the backend and holds its start. It throws when the backend does not stop. */
  stopBackend: () => Promise<void>
  /** Releases the hold and starts the backend with the saved folder. */
  startBackend: () => Promise<void>
  saveAssetsDir: (dir: string | undefined) => void
  emit: (patch: MoveStatePatch) => void
  isCancelled: () => boolean
  logError: (message: string) => void
}

export type MoveRequest = {
  plan: MovePlan
  toCustom: boolean
  /** The custom folder in use now. `undefined` when the default folder is in use. */
  previousDir: string | undefined
  /** Plans the move again. It runs after the backend stopped. */
  replan: () => Promise<PreflightResult>
}

export type MoveOutcome = 'moved' | 'cancelled' | 'failed'

const PROGRESS_INTERVAL_MS = 150

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/** Runs the move. The setting changes only after a verified copy. Every failure ends on the old folder. */
export async function runMove(
  { plan: confirmed, toCustom, previousDir, replan }: MoveRequest,
  deps: MoveDeps,
): Promise<MoveOutcome> {
  const done = (ok: boolean, message: string) =>
    deps.emit({ phase: 'done', canCancel: false, result: { ok, message } })
  const restartNote = (message: string, failure: string | null) =>
    failure ? `${message} The server could not restart: ${failure}` : message
  const restart = async (): Promise<string | null> => {
    try {
      await deps.startBackend()
      return null
    } catch (error) {
      return errorText(error)
    }
  }

  deps.emit({ phase: 'stopping', filesTotal: confirmed.files.length, bytesTotal: confirmed.totalBytes })
  try {
    await deps.stopBackend()
  } catch (error) {
    // The old backend may still run. Do not start a second one on the same database.
    done(false, `The app server did not stop (${errorText(error)}). Restart the app. Your assets are still in the old folder.`)
    return 'failed'
  }

  let created: string[] = []
  let markerCreated = false
  let plan = confirmed
  try {
    // The user read the confirmation some time ago. A file may have been added since then.
    // Plan again now that the backend is stopped, so no file stays behind in the old folder.
    const fresh = await replan()
    if (!fresh.ok) {
      done(false, restartNote(`${fresh.reason} Your assets are still in the old folder.`, await restart()))
      return 'failed'
    }
    plan = fresh.plan

    deps.emit({ phase: 'copying', filesTotal: plan.files.length, bytesTotal: plan.totalBytes })
    let lastEmit = 0
    const copy = await copyFiles(plan, {
      isCancelled: deps.isCancelled,
      onProgress: ({ filesDone, bytesDone }) => {
        const now = Date.now()
        if (now - lastEmit < PROGRESS_INTERVAL_MS && filesDone < plan.files.length) return
        lastEmit = now
        deps.emit({ phase: 'copying', filesDone, bytesDone })
      },
    })
    created = copy.created

    // The commit boundary. A cancel is honored up to here, also when it came in while the
    // last file was copied or while no file needed a copy. There is no await between this
    // check and the change of `canCancel`, so a later cancel is ignored and cannot be lost.
    if (copy.status === 'cancelled' || deps.isCancelled()) {
      await removeCreated(created)
      created = []
      deps.emit({ phase: 'restarting', canCancel: false })
      const failure = await restart()
      done(false, restartNote('The move was cancelled. Your assets are still in the old folder.', failure))
      return failure === null ? 'cancelled' : 'failed'
    }
    deps.emit({ phase: 'restarting', canCancel: false })

    if (toCustom) markerCreated = await writeMarker(plan.to)
    deps.saveAssetsDir(toCustom ? plan.to : undefined)
    const failure = await restart()
    if (failure !== null) throw new Error(`The server did not start with the new folder (${failure}).`)
  } catch (error) {
    deps.logError(`Assets move failed: ${errorText(error)}`)
    await removeCreated(created)
    if (markerCreated) await removeMarker(plan.to)
    deps.saveAssetsDir(previousDir)
    const revert = await restart()
    const reason = errorText(error).replace(/\.$/, '')
    done(false, restartNote(`The move failed: ${reason}. Your assets are still in the old folder.`, revert))
    return 'failed'
  }

  deps.emit({ phase: 'cleaning', canCancel: false })
  const undeleted = await deleteMovedFiles(plan)
  if (undeleted.length === 0 && previousDir) await releaseFolder(plan.from)
  done(
    true,
    undeleted.length === 0
      ? 'Your assets are in the new folder.'
      : `Your assets are in the new folder. The app could not delete ${undeleted.length} old files in ${plan.from}. You can delete them by hand.`,
  )
  return 'moved'
}
