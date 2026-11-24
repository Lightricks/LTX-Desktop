import fs from 'fs'
import fsp from 'fs/promises'
import path from 'path'
import { formatBytes } from '../shared/format-bytes.ts'
import { isRemoteFilePath } from '../shared/remote-file-path.ts'

/** Same name as ASSETS_MARKER_FILENAME in backend/services/sqlite_store.py. */
export const ASSETS_MARKER_FILENAME = '.ltx-assets'
// Only files are moved. The backend's `.in-flight` folder stays: the backend clears it at boot.
const PARTIAL_SUFFIX = '.ltx-partial'
const COPY_CHUNK_BYTES = 4 * 1024 * 1024
// Same name as IN_FLIGHT_DIRNAME in backend/runtime_config/assets_layout.py.
const IN_FLIGHT_DIRNAME = '.in-flight'

export type SourceFile = { name: string; size: number }

export type MovePlan = {
  from: string
  to: string
  /** Every asset file in the old folder (see ASSET_FILE_NAME). They are all deleted after the move. */
  files: SourceFile[]
  totalBytes: number
  freeBytes: number
}

export type PreflightResult = { ok: true; plan: MovePlan } | { ok: false; reason: string }

export type PreflightOptions = {
  from: string
  to: string
  /** The target equals or contains one of these folders: reject. */
  protectedDirs: string[]
  /** The target is inside one of these system folders: reject. */
  blockedRoots: string[]
  /** The target is inside one of these app folders (app data, install): reject. */
  appFolders: string[]
  freeBytes: (dir: string) => Promise<number>
}

export type MoveProgress = { filesDone: number; bytesDone: number }

export type CopyHooks = {
  onProgress: (progress: MoveProgress) => void
  isCancelled: () => boolean
}

export type CopyResult = { status: 'done' | 'cancelled'; created: string[] }

/**
 * The real path. A folder that does not exist yet is resolved through its nearest existing parent,
 * so a new folder behind a link compares the same way as an existing one.
 */
function realpathOrResolve(p: string): string {
  const missing: string[] = []
  let current = path.resolve(p)
  for (;;) {
    try {
      return path.join(fs.realpathSync.native(current), ...missing.reverse())
    } catch {
      const parent = path.dirname(current)
      if (parent === current) return path.resolve(p)
      missing.push(path.basename(current))
      current = parent
    }
  }
}

function comparable(p: string): string {
  const resolved = realpathOrResolve(p)
  return process.platform === 'win32' ? resolved.toLowerCase() : resolved
}

/** True when `child` is `parent` or is inside it. */
function isInside(child: string, parent: string): boolean {
  const relative = path.relative(comparable(parent), comparable(child))
  return (
    relative === '' ||
    (relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative))
  )
}

// The store names its files `<uuid4><ext>` and `<uuid4>-thumb.jpg` (see _derived_path and
// ThumbnailWriter in the backend). Only those files are moved or deleted. A custom folder can
// hold the user's own files, and the app must never touch them.
const ASSET_FILE_NAME = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(-thumb)?\.[a-z0-9]+$/i

async function listSourceFiles(dir: string): Promise<SourceFile[]> {
  const files: SourceFile[] = []
  for (const name of await fsp.readdir(dir)) {
    if (!ASSET_FILE_NAME.test(name)) continue
    // stat follows a link, as the backend does. The copy holds the content, and the delete removes only the link.
    const stat = await fsp.stat(path.join(dir, name)).catch(() => null)
    if (stat?.isFile()) files.push({ name, size: stat.size })
  }
  return files
}

function rejectTarget(options: PreflightOptions): string | null {
  const { from, to } = options
  if (isRemoteFilePath(to)) return 'Network folders are not supported. Choose a folder on a local drive.'
  if (path.parse(path.resolve(to)).root === path.resolve(to)) return 'Choose a folder, not a drive root.'
  if (comparable(from) === comparable(to)) return 'This is the folder that the app uses now.'
  if (isInside(to, from) || isInside(from, to)) {
    return 'The new folder and the current folder must not contain each other.'
  }
  if (options.protectedDirs.some((dir) => isInside(dir, to))) {
    return 'This folder holds app or personal data. Choose another folder.'
  }
  if (options.blockedRoots.some((dir) => isInside(to, dir))) {
    return 'This is a system folder. Choose another folder.'
  }
  if (options.appFolders.some((dir) => isInside(to, dir))) {
    return 'This folder is inside the app folders. Choose another folder.'
  }
  return null
}

async function probeWritable(dir: string): Promise<boolean> {
  const probe = path.join(dir, `.ltx-probe-${process.pid}-${Date.now()}`)
  try {
    await fsp.writeFile(probe, '')
    await fsp.unlink(probe)
    return true
  } catch {
    return false
  }
}

/** Checks the move and changes nothing, except that a missing target folder is created. */
export async function preflightMove(options: PreflightOptions): Promise<PreflightResult> {
  const { from, to } = options
  const rejected = rejectTarget(options)
  if (rejected) return { ok: false, reason: rejected }

  try {
    await fsp.mkdir(to, { recursive: true })
    if (!(await fsp.stat(to)).isDirectory()) return { ok: false, reason: 'The path is not a folder.' }
  } catch {
    return { ok: false, reason: 'The folder cannot be created or read.' }
  }
  if (!(await probeWritable(to))) {
    return { ok: false, reason: 'The app has no write access to this folder.' }
  }

  let files: SourceFile[]
  try {
    files = await listSourceFiles(from)
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
      return { ok: false, reason: 'The current assets folder was not found. Connect it first.' }
    }
    throw err
  }
  const totalBytes = files.reduce((sum, file) => sum + file.size, 0)
  const freeBytes = await options.freeBytes(to)
  if (totalBytes > freeBytes) {
    return {
      ok: false,
      reason: `Not enough space: ${formatBytes(totalBytes)} needed, ${formatBytes(freeBytes)} free.`,
    }
  }

  return {
    ok: true,
    plan: {
      from,
      to,
      files,
      totalBytes,
      freeBytes,
    },
  }
}

class CancelledError extends Error {}

async function copyOne(
  source: string,
  partial: string,
  onBytes: (bytes: number) => void,
  isCancelled: () => boolean,
): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const input = fs.createReadStream(source, { highWaterMark: COPY_CHUNK_BYTES })
    const output = fs.createWriteStream(partial)
    // Reject only after the output handle is closed. On Windows an open handle blocks the
    // removal of the partial file.
    const fail = (err: Error) => {
      input.destroy()
      if (output.closed) reject(err)
      else {
        output.once('close', () => reject(err))
        output.destroy()
      }
    }
    input.on('data', (chunk) => {
      onBytes(chunk.length)
      if (isCancelled()) input.destroy(new CancelledError())
    })
    input.on('error', fail)
    output.on('error', fail)
    // 'close', not 'finish': on Windows the file cannot be renamed before the handle is closed.
    output.on('close', resolve)
    input.pipe(output)
  })
}

async function removeQuietly(file: string): Promise<void> {
  await fsp.rm(file, { force: true }).catch(() => undefined)
}

/**
 * Copies every file. A file with the same name in the target is replaced. Each file goes to a partial name first,
 * its size is checked, and then it gets its final name. A cancel or an error removes
 * everything this call created.
 */
export async function copyFiles(plan: MovePlan, hooks: CopyHooks): Promise<CopyResult> {
  const created: string[] = []
  let filesDone = 0
  let bytesDone = 0
  const cleanup = async () => {
    for (const file of created) await removeQuietly(file)
  }
  try {
    for (const file of plan.files) {
      if (hooks.isCancelled()) throw new CancelledError()
      const target = path.join(plan.to, file.name)
      const partial = target + PARTIAL_SUFFIX
      created.push(partial)
      await copyOne(
        path.join(plan.from, file.name),
        partial,
        (bytes) => {
          bytesDone += bytes
          hooks.onProgress({ filesDone, bytesDone })
        },
        hooks.isCancelled,
      )
      if ((await fsp.stat(partial)).size !== file.size) {
        throw new Error(`The copy of ${file.name} has the wrong size.`)
      }
      // The old file is deleted later. Flush the copy to disk first, so a power loss cannot
      // leave a truncated copy next to a deleted original.
      const handle = await fsp.open(partial, 'r+')
      try {
        await handle.sync()
      } finally {
        await handle.close()
      }
      // A file that was in the target before stays when a later step fails: it is not ours to delete.
      const existed = await fsp.access(target).then(() => true, () => false)
      await fsp.rename(partial, target)
      if (existed) created.pop()
      else created[created.length - 1] = target
      filesDone += 1
      hooks.onProgress({ filesDone, bytesDone })
    }
    return { status: 'done', created }
  } catch (err) {
    await cleanup()
    if (err instanceof CancelledError || hooks.isCancelled()) return { status: 'cancelled', created: [] }
    throw err
  }
}

/** The marker tells the backend that this folder is the library, not an empty mount point. Returns false when it was there already. */
export async function writeMarker(dir: string): Promise<boolean> {
  try {
    await fsp.writeFile(path.join(dir, ASSETS_MARKER_FILENAME), '', { flag: 'wx' })
    return true
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'EEXIST') return false
    throw err
  }
}

export async function removeMarker(dir: string): Promise<void> {
  await fsp.rm(path.join(dir, ASSETS_MARKER_FILENAME), { force: true }).catch(() => undefined)
}

/** Removes the marker, and the in-flight folder if it is empty, from a folder the app left. */
export async function releaseFolder(dir: string): Promise<void> {
  await removeMarker(dir)
  await fsp.rmdir(path.join(dir, IN_FLIGHT_DIRNAME)).catch(() => undefined)
}

/** Removes files that a failed or reverted move created in the new folder. */
export async function removeCreated(created: string[]): Promise<void> {
  for (const file of created) await removeQuietly(file)
}

/** Deletes the moved files from the old folder. Returns the files that could not be deleted. */
export async function deleteMovedFiles(plan: MovePlan): Promise<string[]> {
  const failed: string[] = []
  for (const file of plan.files) {
    const full = path.join(plan.from, file.name)
    try {
      await fsp.rm(full, { force: true })
    } catch {
      failed.push(full)
    }
  }
  return failed
}

export type AssetsDirCheck = { ok: true } | { ok: false; reason: string }

/** Startup check for a saved custom folder: it must exist, be writable and hold the marker. */
export async function checkCustomAssetsDir(dir: string): Promise<AssetsDirCheck> {
  // A network path would make Windows send credentials to the remote host.
  if (isRemoteFilePath(dir)) return { ok: false, reason: 'Network folders are not supported.' }
  try {
    if (!(await fsp.stat(dir)).isDirectory()) return { ok: false, reason: 'The path is not a folder.' }
  } catch {
    return { ok: false, reason: 'The folder was not found.' }
  }
  const hasMarker = await fsp.access(path.join(dir, ASSETS_MARKER_FILENAME)).then(() => true, () => false)
  if (!hasMarker) {
    return { ok: false, reason: `The folder has no ${ASSETS_MARKER_FILENAME} file. It may be an empty mount point.` }
  }
  if (!(await probeWritable(dir))) return { ok: false, reason: 'The app has no write access to the folder.' }
  return { ok: true }
}
