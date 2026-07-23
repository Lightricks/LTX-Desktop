import { execFile } from 'child_process'
import { app } from 'electron'
import fs from 'fs'
import http from 'http'
import https from 'https'
import { load as loadYaml } from 'js-yaml'
import path from 'path'
import { isDev } from './config'
import { logger } from './logger'

export interface PythonSetupProgress {
  status: 'downloading' | 'extracting' | 'complete' | 'error'
  percent: number
  downloadedBytes: number
  totalBytes: number
  speed: number
}

interface ArchiveManifest {
  parts: { name: string; size: number }[]
  totalSize: number
}

// ── GitHub private repo authentication ────────────────────────────────
// Mirrors electron-updater: only sends GH_TOKEN when `private: true` is set
// in the publish config (app-update.yml). This prevents accidental token leaks
// for public repos.

let _authHeaders: Record<string, string> | null = null

function getAuthHeaders(): Record<string, string> {
  if (_authHeaders !== null) return _authHeaders

  _authHeaders = {}

  const configPath = isDev
    ? path.join(process.cwd(), 'dev-app-update.yml')
    : path.join(process.resourcesPath, 'app-update.yml')

  let isPrivate = false
  try {
    const config = loadYaml(fs.readFileSync(configPath, 'utf-8')) as Record<string, unknown>
    isPrivate = config?.private === true
  } catch { /* no config file — public repo */ }

  if (isPrivate) {
    const token = process.env.GH_TOKEN || process.env.GITHUB_TOKEN
    if (token) {
      _authHeaders = { authorization: `token ${token}` }
    }
  }

  return _authHeaders
}

function getBundledHashPath(): string {
  if (isDev) {
    return path.join(process.cwd(), 'python-deps-hash.txt')
  }
  return path.join(process.resourcesPath, 'python-deps-hash.txt')
}

function getInstalledHashPath(): string {
  return path.join(app.getPath('userData'), 'python', 'deps-hash.txt')
}

/** Directory where python-embed lives at runtime. */
export function getPythonDir(): string {
  if (process.platform === 'win32' || process.platform === 'linux') {
    if (isDev) {
      return path.join(process.cwd(), 'python-embed')
    }
    return path.join(app.getPath('userData'), 'python')
  }
  // macOS: bundled in resources
  return path.join(process.resourcesPath, 'python')
}

/**
 * Check whether the Python environment is ready to use.
 * Also promotes a staged python-next/ directory if it matches the expected hash.
 */
export function isPythonReady(): { ready: boolean } {
  if (process.platform === 'darwin') {
    return { ready: true }
  }

  if (isDev) {
    return { ready: true }
  }

  const bundledHash = readHash(getBundledHashPath())

  // Check if a pre-downloaded python-next/ is waiting to be promoted
  const nextDir = path.join(app.getPath('userData'), 'python-next')
  const nextHash = readHash(path.join(nextDir, 'deps-hash.txt'))
  if (bundledHash && nextHash && bundledHash === nextHash) {
    logger.info( '[python-setup] Promoting staged python-next/ to python/')
    try {
      const destDir = path.join(app.getPath('userData'), 'python')
      if (fs.existsSync(destDir)) {
        fs.rmSync(destDir, { recursive: true, force: true })
      }
      fs.renameSync(nextDir, destDir)
      return { ready: true }
    } catch (err) {
      logger.error( `[python-setup] Failed to promote staged python: ${err}`)
      // Fall through to normal check
    }
  }

  const installedHash = readHash(getInstalledHashPath())

  if (!bundledHash) {
    const pythonExe = path.join(getPythonDir(), process.platform === 'win32' ? 'python.exe' : 'bin/python3')
    return { ready: fs.existsSync(pythonExe) }
  }

  return { ready: bundledHash === installedHash }
}

/**
 * Pre-download python-embed for an upcoming app update (Windows only).
 * Downloads to userData/python-next/ so the next launch can promote it instantly.
 * Returns true if a download was performed, false if not needed.
 */
export async function preDownloadPythonForUpdate(
  newVersion: string,
  onProgress?: (progress: PythonSetupProgress) => void
): Promise<boolean> {
  if (process.platform === 'darwin') {
    return false
  }

  const baseUrl = (isDev && process.env.LTX_PYTHON_URL?.replace(/^["']+|["']+$/g, ''))
    || `https://github.com/Lightricks/ltx-desktop/releases/download/v${newVersion}`

  // Fetch the new version's deps hash
  const newHash = await fetchRemoteDepsHash(baseUrl, 'python-next-hash-check.txt')

  if (!newHash) {
    logger.info( '[python-setup] No deps hash available for new version, skipping pre-download')
    return false
  }

  // Compare with currently installed hash
  const installedHash = readHash(getInstalledHashPath())
  if (newHash === installedHash) {
    logger.info( '[python-setup] Python deps unchanged in new version, no pre-download needed')
    return false
  }

  logger.info( `[python-setup] Python deps changed (${installedHash} → ${newHash}), pre-downloading`)

  // Download to python-next/
  const nextDir = path.join(app.getPath('userData'), 'python-next')
  const tempDir = path.join(app.getPath('userData'), 'python-next-tmp')
  const archivePath = path.join(app.getPath('userData'), 'python-next.tar.gz')

  try {
    if (fs.existsSync(nextDir)) fs.rmSync(nextDir, { recursive: true, force: true })
    if (fs.existsSync(tempDir)) fs.rmSync(tempDir, { recursive: true, force: true })
  } catch { /* ignore */ }

  fs.mkdirSync(tempDir, { recursive: true })

  const cleanupFiles: string[] = []
  const noop = () => {}
  const progressCb = onProgress || noop

  try {
    try {
      await acquireArchive(baseUrl, archivePath, cleanupFiles, progressCb)
    } catch (primaryErr) {
      const prefix = getPythonArchivePrefix()
      const fallbackUrl = newHash ? `${FALLBACK_CDN_BASE}/${prefix}/${newHash}/${prefix}.tar.gz` : null
      if (!fallbackUrl || isLocalPath(baseUrl)) {
        throw primaryErr
      }
      logger.warn( `[python-setup] Pre-download primary failed: ${primaryErr}`)
      logger.info( `[python-setup] Falling back to CDN: ${fallbackUrl}`)
      try { fs.unlinkSync(archivePath) } catch { /* ignore */ }
      for (const f of cleanupFiles) { try { fs.unlinkSync(f) } catch { /* ignore */ } }
      cleanupFiles.length = 0
      await acquireArchive(fallbackUrl, archivePath, cleanupFiles, progressCb)
    }

    await extractTarGz(archivePath, tempDir)

    const extractedInner = path.join(tempDir, 'python-embed')
    const extractedSource = fs.existsSync(extractedInner) ? extractedInner : tempDir

    if (fs.existsSync(nextDir)) fs.rmSync(nextDir, { recursive: true, force: true })
    fs.renameSync(extractedSource, nextDir)

    // Write the new hash into python-next/ so isPythonReady can verify it on next launch
    fs.writeFileSync(path.join(nextDir, 'deps-hash.txt'), newHash)

    logger.info( '[python-setup] Pre-download complete, staged at python-next/')
    return true
  } catch (err) {
    logger.error( `[python-setup] Pre-download failed: ${err}`)
    try { fs.rmSync(nextDir, { recursive: true, force: true }) } catch { /* ignore */ }
    return false
  } finally {
    try { fs.unlinkSync(archivePath) } catch { /* ignore */ }
    for (const f of cleanupFiles) {
      try { fs.unlinkSync(f) } catch { /* ignore */ }
    }
    try { if (fs.existsSync(tempDir)) fs.rmSync(tempDir, { recursive: true, force: true }) } catch { /* ignore */ }
  }
}

function readHash(filePath: string): string | null {
  try {
    return fs.readFileSync(filePath, 'utf-8').trim()
  } catch {
    return null
  }
}

// ── Archive source resolution ─────────────────────────────────────────
// Primary: GitHub Releases (multi-part, version-based)
// Fallback: public CDN bucket (single file, deps-hash-based)

const FALLBACK_CDN_BASE = 'https://storage.googleapis.com/ltx-desktop-artifacts'

function getPythonArchivePrefix(): string {
  if (process.platform === 'win32') return 'python-embed-win32'
  if (process.platform === 'linux') {
    if (process.arch === 'x64') return 'python-embed-linux-x64'
    if (process.arch === 'arm64') return 'python-embed-linux-arm64'
    throw new Error(`Unsupported Linux architecture: ${process.arch}`)
  }
  throw new Error(`Python download is not supported on ${process.platform}`)
}

function getArchiveBase(): string {
  // LTX_PYTHON_URL is a dev-only override for testing with local archives.
  // Disabled in production to prevent code injection into a signed app.
  if (isDev && process.env.LTX_PYTHON_URL) {
    return process.env.LTX_PYTHON_URL.replace(/^["']+|["']+$/g, '')
  }
  const version = app.getVersion()
  return `https://github.com/Lightricks/ltx-desktop/releases/download/v${version}`
}

function getFallbackArchiveUrl(): string | null {
  const hash = readHash(getBundledHashPath())
  if (!hash) return null
  const prefix = getPythonArchivePrefix()
  return `${FALLBACK_CDN_BASE}/${prefix}/${hash}/${prefix}.tar.gz`
}

function isLocalPath(source: string): boolean {
  return !source.startsWith('http://') && !source.startsWith('https://')
}

/**
 * Fetch the deps-hash for a python-embed source (local dir/archive or remote base URL),
 * without downloading the full archive. Returns null if unavailable.
 */
async function fetchRemoteDepsHash(base: string, tempFileName: string): Promise<string | null> {
  if (isLocalPath(base)) {
    if (base.endsWith('.tar.gz')) return null // can't inspect a single tar.gz without extracting
    return readHash(path.join(base, 'deps-hash.txt'))
  }
  const hashUrl = `${base}/python-deps-hash.txt`
  const hashDest = path.join(app.getPath('userData'), tempFileName)
  try {
    await downloadFileRaw(hashUrl, hashDest)
    return readHash(hashDest)
  } catch (err) {
    logger.warn( `[python-setup] Could not fetch deps hash from ${hashUrl}: ${err}`)
    return null
  } finally {
    try { fs.unlinkSync(hashDest) } catch { /* ignore */ }
  }
}

/**
 * Acquire the python-embed archive from a source (local, GitHub, or CDN).
 * Returns once the archive is written to archivePath.
 */
async function acquireArchive(
  base: string,
  archivePath: string,
  cleanupFiles: string[],
  onProgress: (progress: PythonSetupProgress) => void
): Promise<void> {
  if (isLocalPath(base) && base.endsWith('.tar.gz')) {
    await copyFileWithProgress(base, archivePath, 0, fs.statSync(base).size, onProgress)
  } else if (isLocalPath(base)) {
    await acquirePartsLocal(base, archivePath, cleanupFiles, onProgress)
  } else if (base.includes('/releases/download/')) {
    // GitHub Releases — multi-part
    await acquirePartsRemote(base, archivePath, cleanupFiles, onProgress)
  } else {
    // CDN or other URL — single file (content-length discovered from response)
    let lastTime = Date.now()
    let lastBytes = 0
    let speed = 0

    await downloadFileWithGlobalProgress(base, archivePath, 0, 0, (downloaded, totalBytes) => {
      const now = Date.now()
      const elapsed = (now - lastTime) / 1000
      if (elapsed >= 1) {
        speed = (downloaded - lastBytes) / elapsed
        lastTime = now
        lastBytes = downloaded
      }

      onProgress({
        status: 'downloading',
        percent: totalBytes > 0 ? Math.round((downloaded / totalBytes) * 100) : 0,
        downloadedBytes: downloaded,
        totalBytes,
        speed,
      })
    })
  }
}

/**
 * Download (or copy) python-embed archive and extract to userData/python/.
 * Tries GitHub Releases first, falls back to CDN if available.
 */
export async function downloadPythonEmbed(
  onProgress: (progress: PythonSetupProgress) => void
): Promise<void> {
  const destDir = path.join(app.getPath('userData'), 'python')
  const tempDir = path.join(app.getPath('userData'), 'python-tmp')
  const archivePath = path.join(app.getPath('userData'), `${getPythonArchivePrefix()}.tar.gz`)

  try {
    if (fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true })
    }
  } catch { /* ignore */ }

  fs.mkdirSync(tempDir, { recursive: true })

  const cleanupFiles: string[] = []

  try {
    const base = getArchiveBase()
    logger.info( `[python-setup] Archive base: ${base}`)

    // Verify the source's deps hash matches what this build expects before downloading
    // the (multi-GB) archive. A version-tagged release can be stale relative to the
    // currently bundled backend code (e.g. deps changed without a version bump), so a
    // successful download from `base` is not by itself proof the content is correct.
    const expectedHash = readHash(getBundledHashPath())
    let source = base
    if (expectedHash) {
      const remoteHash = await fetchRemoteDepsHash(base, `${getPythonArchivePrefix()}-hash-check.txt`)
      if (remoteHash === null && isLocalPath(base)) {
        // A local archive (dev/testing) has no sidecar hash to fetch, so the check is
        // unverifiable rather than mismatched — skip it instead of hard-failing, which
        // otherwise makes a valid local python-embed.tar.gz unusable whenever a bundled
        // hash is present.
        logger.warn(`[python-setup] Local archive at ${base}: no remote deps hash to verify, skipping check`)
      } else if (remoteHash !== expectedHash) {
        const fallbackUrl = getFallbackArchiveUrl()
        if (!fallbackUrl || isLocalPath(base)) {
          throw new Error(
            `Python deps hash mismatch at ${base} (expected ${expectedHash}, got ${remoteHash ?? 'unknown'}) and no CDN fallback available`
          )
        }
        logger.warn(
          `[python-setup] Archive at ${base} has deps hash ${remoteHash ?? 'unknown'}, expected ${expectedHash} — using hash-addressed CDN instead`
        )
        source = fallbackUrl
      }
    } else {
      logger.warn( '[python-setup] No bundled deps hash found, skipping archive verification')
    }

    try {
      await acquireArchive(source, archivePath, cleanupFiles, onProgress)
    } catch (primaryErr) {
      // Primary source failed — try CDN fallback
      const fallbackUrl = getFallbackArchiveUrl()
      if (!fallbackUrl || isLocalPath(source) || source === fallbackUrl) {
        throw primaryErr
      }

      logger.warn( `[python-setup] Primary download failed: ${primaryErr}`)
      logger.info( `[python-setup] Falling back to CDN: ${fallbackUrl}`)

      // Clean up any partial primary download
      try { fs.unlinkSync(archivePath) } catch { /* ignore */ }
      for (const f of cleanupFiles) {
        try { fs.unlinkSync(f) } catch { /* ignore */ }
      }
      cleanupFiles.length = 0

      await acquireArchive(fallbackUrl, archivePath, cleanupFiles, onProgress)
    }

    // Extract
    onProgress({ status: 'extracting', percent: 100, downloadedBytes: 0, totalBytes: 0, speed: 0 })
    logger.info( `[python-setup] Extracting to: ${tempDir}`)
    await extractTarGz(archivePath, tempDir)

    // Move into place (archive has top-level `python-embed/` directory)
    const extractedInner = path.join(tempDir, 'python-embed')
    const extractedSource = fs.existsSync(extractedInner) ? extractedInner : tempDir

    if (fs.existsSync(destDir)) {
      fs.rmSync(destDir, { recursive: true, force: true })
    }
    fs.renameSync(extractedSource, destDir)

    // Write deps hash so subsequent launches skip download
    const bundledHash = getBundledHashPath()
    if (fs.existsSync(bundledHash)) {
      fs.copyFileSync(bundledHash, path.join(destDir, 'deps-hash.txt'))
    }

    onProgress({ status: 'complete', percent: 100, downloadedBytes: 0, totalBytes: 0, speed: 0 })
    logger.info( '[python-setup] Python environment ready')
  } catch (err) {
    try { fs.rmSync(tempDir, { recursive: true, force: true }) } catch { /* ignore */ }
    try { fs.rmSync(destDir, { recursive: true, force: true }) } catch { /* ignore */ }
    throw err
  } finally {
    try { fs.unlinkSync(archivePath) } catch { /* ignore */ }
    for (const f of cleanupFiles) {
      try { fs.unlinkSync(f) } catch { /* ignore */ }
    }
    try { if (fs.existsSync(tempDir)) fs.rmSync(tempDir, { recursive: true, force: true }) } catch { /* ignore */ }
  }
}

// ── Multi-part: local directory ──────────────────────────────────────

async function acquirePartsLocal(
  dirPath: string,
  archivePath: string,
  cleanupFiles: string[],
  onProgress: (progress: PythonSetupProgress) => void
): Promise<void> {
  const manifestPath = path.join(dirPath, `${getPythonArchivePrefix()}.manifest.json`)
  const manifest: ArchiveManifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'))

  const partPaths: string[] = []
  let bytesSoFar = 0

  for (const part of manifest.parts) {
    const src = path.join(dirPath, part.name)
    const dest = path.join(app.getPath('userData'), part.name)
    partPaths.push(dest)
    cleanupFiles.push(dest)

    await copyFileWithProgress(src, dest, bytesSoFar, manifest.totalSize, onProgress)
    bytesSoFar += part.size
  }

  await concatenateParts(partPaths, archivePath)
}

// ── Multi-part: remote download ──────────────────────────────────────

async function acquirePartsRemote(
  baseUrl: string,
  archivePath: string,
  cleanupFiles: string[],
  onProgress: (progress: PythonSetupProgress) => void
): Promise<void> {
  // Fetch manifest
  const prefix = getPythonArchivePrefix()
  const manifestUrl = `${baseUrl}/${prefix}.manifest.json`
  const manifestDest = path.join(app.getPath('userData'), `${prefix}.manifest.json`)
  cleanupFiles.push(manifestDest)
  await downloadFileRaw(manifestUrl, manifestDest)
  const manifest: ArchiveManifest = JSON.parse(fs.readFileSync(manifestDest, 'utf-8'))

  const partPaths: string[] = []
  let bytesSoFar = 0
  let lastTime = Date.now()
  let lastReportedBytes = 0
  let speed = 0

  for (const part of manifest.parts) {
    const partUrl = `${baseUrl}/${part.name}`
    const partDest = path.join(app.getPath('userData'), part.name)
    partPaths.push(partDest)
    cleanupFiles.push(partDest)

    await downloadFileWithGlobalProgress(
      partUrl,
      partDest,
      bytesSoFar,
      manifest.totalSize,
      (globalDownloaded, totalBytes) => {
        const now = Date.now()
        const elapsed = (now - lastTime) / 1000

        if (elapsed >= 1) {
          speed = (globalDownloaded - lastReportedBytes) / elapsed
          lastTime = now
          lastReportedBytes = globalDownloaded
        }

        onProgress({
          status: 'downloading',
          percent: Math.round((globalDownloaded / totalBytes) * 100),
          downloadedBytes: globalDownloaded,
          totalBytes,
          speed,
        })
      }
    )

    bytesSoFar += part.size
  }

  await concatenateParts(partPaths, archivePath)
}

// ── File operations ──────────────────────────────────────────────────

function concatenateParts(parts: string[], dest: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const writeStream = fs.createWriteStream(dest)
    let i = 0

    function writeNext() {
      if (i >= parts.length) {
        writeStream.end(() => resolve())
        return
      }

      const readStream = fs.createReadStream(parts[i])
      i++

      readStream.on('error', (err) => {
        writeStream.destroy()
        reject(err)
      })

      readStream.on('end', writeNext)
      readStream.pipe(writeStream, { end: false })
    }

    writeStream.on('error', reject)
    writeNext()
  })
}

/** Copy a local file with progress relative to a global total. */
function copyFileWithProgress(
  source: string,
  dest: string,
  globalOffset: number,
  globalTotal: number,
  onProgress: (progress: PythonSetupProgress) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    let copiedBytes = 0

    const readStream = fs.createReadStream(source)
    const writeStream = fs.createWriteStream(dest)

    readStream.on('data', (chunk: Buffer) => {
      copiedBytes += chunk.length
      const totalDone = globalOffset + copiedBytes
      onProgress({
        status: 'downloading',
        percent: Math.round((totalDone / globalTotal) * 100),
        downloadedBytes: totalDone,
        totalBytes: globalTotal,
        speed: 0,
      })
    })

    readStream.on('error', reject)
    writeStream.on('error', reject)
    writeStream.on('finish', resolve)

    readStream.pipe(writeStream)
  })
}

/** Download a file without progress (used for manifest). */
function downloadFileRaw(url: string, dest: string, redirectCount = 0): Promise<void> {
  return new Promise((resolve, reject) => {
    if (redirectCount > 5) {
      reject(new Error('Too many redirects'))
      return
    }

    const client = url.startsWith('https') ? https : http
    const req = client.get(url, { headers: getAuthHeaders() }, (res) => {
      if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume()
        downloadFileRaw(res.headers.location, dest, redirectCount + 1).then(resolve).catch(reject)
        return
      }
      if (!res.statusCode || res.statusCode >= 400) {
        res.resume()
        reject(new Error(`Download failed: HTTP ${res.statusCode}`))
        return
      }

      const file = fs.createWriteStream(dest)
      res.pipe(file)
      file.on('finish', () => file.close(() => resolve()))
      file.on('error', (err) => { fs.unlink(dest, () => {}); reject(err) })
    })

    req.on('error', reject)
  })
}

/** Download a file, reporting progress as (globalDownloaded, globalTotal). */
function downloadFileWithGlobalProgress(
  url: string,
  dest: string,
  globalOffset: number,
  globalTotal: number,
  onProgress: (globalDownloaded: number, globalTotal: number) => void,
  redirectCount = 0
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (redirectCount > 5) {
      reject(new Error('Too many redirects'))
      return
    }

    const client = url.startsWith('https') ? https : http
    const req = client.get(url, { headers: getAuthHeaders() }, (res) => {
      if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume()
        downloadFileWithGlobalProgress(res.headers.location, dest, globalOffset, globalTotal, onProgress, redirectCount + 1)
          .then(resolve).catch(reject)
        return
      }
      if (!res.statusCode || res.statusCode >= 400) {
        res.resume()
        reject(new Error(`Download failed: HTTP ${res.statusCode}`))
        return
      }

      // If caller didn't know total, use content-length from response
      const effectiveTotal = globalTotal || parseInt(res.headers['content-length'] || '0', 10)

      let downloadedBytes = 0
      const file = fs.createWriteStream(dest)
      res.pipe(file)

      res.on('data', (chunk: Buffer) => {
        downloadedBytes += chunk.length
        onProgress(globalOffset + downloadedBytes, effectiveTotal)
      })

      file.on('finish', () => file.close(() => resolve()))
      file.on('error', (err) => { fs.unlink(dest, () => {}); reject(err) })
    })

    req.on('error', reject)
  })
}

/** Extract a .tar.gz file using the system tar command (ships on Windows 10+). */
function extractTarGz(archive: string, destDir: string): Promise<void> {
  return new Promise((resolve, reject) => {
    execFile('tar', ['-xzf', archive, '-C', destDir], (err, _stdout, stderr) => {
      if (err) {
        reject(new Error(`tar extraction failed: ${stderr || err.message}`))
        return
      }
      resolve()
    })
  })
}
