import { dialog, shell } from 'electron'
import path from 'path'
import fs from 'fs'
import { getAllowedRoots } from '../config'
import { getMainWindow } from '../window'
import { validatePath } from '../path-validation'
import { logger } from '../logger'
import { handle } from './typed-handle'
import { defaultLibraryRoot, resolveLibraryRoot, setLibraryRootOverride } from '../gpm-library-root'

// Real-filesystem media library for the Prompt Manager Pro "Downloads Browser".
// Lives under the OS Downloads dir (an allowed root) so files are real and the
// user can open the folder in their file manager.

const VIDEO_EXT = new Set(['.mp4', '.webm', '.mkv', '.mov', '.avi'])
const AUDIO_EXT = new Set(['.mp3', '.wav', '.ogg', '.aac', '.flac', '.m4a'])
const IMAGE_EXT = new Set(['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'])
const MEDIA_EXT = new Set([...IMAGE_EXT, ...VIDEO_EXT, ...AUDIO_EXT])
const MIME_BY_EXT: Record<string, string> = {
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp',
  '.gif': 'image/gif', '.bmp': 'image/bmp',
  '.mp4': 'video/mp4', '.webm': 'video/webm', '.mkv': 'video/x-matroska', '.mov': 'video/quicktime', '.avi': 'video/x-msvideo',
  '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.ogg': 'audio/ogg', '.aac': 'audio/aac', '.flac': 'audio/flac', '.m4a': 'audio/mp4',
}

function libRoot(): string {
  const root = resolveLibraryRoot()
  fs.mkdirSync(root, { recursive: true })
  return root
}

// Watch the library root so the Studio Assets panel refreshes the moment files
// are added/removed on disk (e.g. a "Save video" into the folder), instead of
// only on manual refresh. Debounced because fs.watch fires several events per
// operation; recursive so changes inside subfolders (Inbox, etc.) are seen.
let libWatcher: fs.FSWatcher | null = null
let watchDebounce: ReturnType<typeof setTimeout> | null = null

function notifyLibChanged(): void {
  if (watchDebounce) clearTimeout(watchDebounce)
  watchDebounce = setTimeout(() => {
    getMainWindow()?.webContents.send('gpm-lib-changed')
  }, 250)
}

function startLibWatch(): void {
  if (libWatcher) { libWatcher.close(); libWatcher = null }
  try {
    libWatcher = fs.watch(libRoot(), { recursive: true }, () => notifyLibChanged())
  } catch (e) {
    // recursive watch is unsupported on some platforms (e.g. Linux) — the panel
    // still works via manual Refresh; just no live updates there.
    logger.warn(`gpm library watch failed: ${e}`)
  }
}

function safeName(name: string): string {
  if (!name || name.includes('/') || name.includes('\\') || name.includes('..')) {
    throw new Error(`Invalid name: ${name}`)
  }
  return name.trim()
}

/** Ensure a named library subfolder exists and return its absolute path. Used by
 *  features that write generated assets straight into the library (e.g. "Continue
 *  as new shot" saving the extracted frame + trimmed clip into Continuations). */
export function ensureLibFolder(name: string): string {
  const p = folderPath(name)
  fs.mkdirSync(p, { recursive: true })
  return p
}

function folderPath(name: string): string {
  const p = path.join(libRoot(), safeName(name))
  validatePath(p, getAllowedRoots())
  return p
}

function uniqueTarget(target: string): string {
  if (!fs.existsSync(target)) return target
  const dir = path.dirname(target)
  const ext = path.extname(target)
  const base = path.basename(target, ext)
  let i = 1
  let candidate = path.join(dir, `${base}-${i}${ext}`)
  while (fs.existsSync(candidate)) { i++; candidate = path.join(dir, `${base}-${i}${ext}`) }
  return candidate
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

// Content-aware auto-naming: normalize a caller-supplied subject to safe filename chars.
function sanitizeSubject(subject: string): string {
  return subject
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48)
    .replace(/-+$/g, '')
}

// Next "<subject>-NN.<ext>" for a folder — the sequence source of truth is the folder itself
// (append-only: existing NN are never renumbered). Scanned per copy so a batch increments.
function nextSequencedName(dir: string, subject: string, ext: string): string {
  let max = 0
  const re = new RegExp(`^${escapeRegExp(subject)}-(\\d+)${escapeRegExp(ext)}$`, 'i')
  try {
    for (const name of fs.readdirSync(dir)) {
      const m = re.exec(name)
      if (m) max = Math.max(max, parseInt(m[1], 10))
    }
  } catch { /* folder is new / unreadable — start the sequence at 01 */ }
  return `${subject}-${String(max + 1).padStart(2, '0')}${ext}`
}

export type LibraryListing = {
  root: string
  folders: string[]
  files: Array<{ folder: string; name: string; path: string; isVideo: boolean; isAudio: boolean; mtimeMs: number }>
}

/** Snapshot of the Studio Assets library (folders + media files). Shared by the
 *  gpmLibList IPC and the RiX MCP server's library search. */
export function listLibrary(): LibraryListing {
  const root = libRoot()
  let folders = fs.readdirSync(root, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name)
  if (folders.length === 0) {
    fs.mkdirSync(path.join(root, 'Inbox'), { recursive: true })
    folders = ['Inbox']
  }
  const files: LibraryListing['files'] = []
  for (const folder of folders) {
    const fp = path.join(root, folder)
    for (const entry of fs.readdirSync(fp, { withFileTypes: true })) {
      if (!entry.isFile()) continue
      const ext = path.extname(entry.name).toLowerCase()
      if (!MEDIA_EXT.has(ext)) continue
      const full = path.join(fp, entry.name)
      files.push({ folder, name: entry.name, path: full, isVideo: VIDEO_EXT.has(ext), isAudio: AUDIO_EXT.has(ext), mtimeMs: fs.statSync(full).mtimeMs })
    }
  }
  return { root, folders, files }
}

export function registerLibraryHandlers(): void {
  handle('gpmLibList', () => listLibrary())

  handle('gpmLibCreateFolder', ({ name }) => {
    try { fs.mkdirSync(folderPath(name)); return { success: true as const } }
    catch (e) { return { success: false as const, error: e instanceof Error ? e.message : 'create failed' } }
  })

  handle('gpmLibRenameFolder', ({ from, to }) => {
    try { fs.renameSync(folderPath(from), folderPath(to)); return { success: true as const } }
    catch (e) { return { success: false as const, error: e instanceof Error ? e.message : 'rename failed' } }
  })

  handle('gpmLibDeleteFolder', ({ name }) => {
    try { fs.rmSync(folderPath(name), { recursive: true, force: true }); return { success: true as const } }
    catch (e) { return { success: false as const, error: e instanceof Error ? e.message : 'delete failed' } }
  })

  handle('gpmLibAddFiles', ({ folder, srcPaths, baseName }) => {
    try {
      const dest = folderPath(folder)
      fs.mkdirSync(dest, { recursive: true }) // robust: quick-save target may have been deleted
      // Content-aware auto-naming: with a subject, land as "<subject>-NN.<ext>" (sequenced per
      // folder); without one (imports / drag-drop), keep the source's original filename.
      const subject = baseName ? sanitizeSubject(baseName) : ''
      let added = 0
      for (const src of srcPaths) {
        if (!fs.existsSync(src) || !fs.statSync(src).isFile()) continue
        const ext = path.extname(src).toLowerCase()
        if (!MEDIA_EXT.has(ext)) continue
        const targetName = subject ? nextSequencedName(dest, subject, ext) : path.basename(src)
        fs.copyFileSync(src, uniqueTarget(path.join(dest, targetName)))
        added++
      }
      return { success: true as const, added }
    } catch (e) {
      return { success: false as const, error: e instanceof Error ? e.message : 'add failed' }
    }
  })

  handle('gpmLibMoveFile', ({ fromFolder, name, toFolder }) => {
    try {
      const src = path.join(folderPath(fromFolder), safeName(name))
      const target = uniqueTarget(path.join(folderPath(toFolder), safeName(name)))
      fs.renameSync(src, target)
      return { success: true as const }
    } catch (e) { return { success: false as const, error: e instanceof Error ? e.message : 'move failed' } }
  })

  handle('gpmLibDeleteFile', ({ folder, name }) => {
    try { fs.rmSync(path.join(folderPath(folder), safeName(name)), { force: true }); return { success: true as const } }
    catch (e) { return { success: false as const, error: e instanceof Error ? e.message : 'delete failed' } }
  })

  handle('gpmLibReveal', ({ folder }) => {
    try { void shell.openPath(folder ? folderPath(folder) : libRoot()); return { success: true as const } }
    catch (e) { logger.warn(`gpmLibReveal failed: ${e}`); return { success: false as const, error: 'reveal failed' } }
  })

  handle('gpmLibReadAsDataUrl', ({ path: p }) => {
    try {
      validatePath(p, getAllowedRoots())
      const ext = path.extname(p).toLowerCase()
      const mime = MIME_BY_EXT[ext] ?? 'application/octet-stream'
      const b64 = fs.readFileSync(p).toString('base64')
      return { success: true as const, dataUrl: `data:${mime};base64,${b64}` }
    } catch (e) {
      return { success: false as const, error: e instanceof Error ? e.message : 'read failed' }
    }
  })

  handle('gpmLibGetRoot', () => ({ root: libRoot(), isDefault: resolveLibraryRoot() === defaultLibraryRoot() }))

  handle('gpmLibChooseRoot', async () => {
    const win = getMainWindow()
    const result = win
      ? await dialog.showOpenDialog(win, { title: 'Choose Downloads Browser folder', properties: ['openDirectory', 'createDirectory'] })
      : await dialog.showOpenDialog({ title: 'Choose Downloads Browser folder', properties: ['openDirectory', 'createDirectory'] })
    if (result.canceled || result.filePaths.length === 0) return { root: null }
    const chosen = result.filePaths[0]
    setLibraryRootOverride(chosen)
    fs.mkdirSync(chosen, { recursive: true })
    startLibWatch() // re-point the watcher at the newly chosen folder
    return { root: chosen }
  })

  handle('gpmLibResetRoot', () => {
    setLibraryRootOverride(null)
    const root = libRoot()
    startLibWatch()
    return { root }
  })

  startLibWatch()
}
