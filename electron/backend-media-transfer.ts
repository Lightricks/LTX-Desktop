import { app } from 'electron'
import crypto from 'crypto'
import fs from 'fs'
import http from 'http'
import https from 'https'
import os from 'os'
import path from 'path'
import { pipeline } from 'stream/promises'
import {
  getAuthToken,
  getBackendConnectionMode,
  getBackendUrl,
} from './python-backend'
import { logger } from './logger'

export type BackendMediaType = 'image' | 'audio' | 'video'

export interface BackendMediaRef {
  media_id: string
  media_type: BackendMediaType
  filename: string
  content_type: string
  size_bytes: number
  sha256: string
  expires_at: string
}

export interface BackendArtifactRef {
  artifact_id: string
  media_type: BackendMediaType
  filename: string
  content_type: string
  size_bytes: number
  sha256: string
  expires_at: string
}

const CONTENT_TYPES: Record<string, string> = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.wav': 'audio/wav',
  '.mp3': 'audio/mpeg',
  '.m4a': 'audio/mp4',
  '.aac': 'audio/aac',
  '.flac': 'audio/flac',
  '.mp4': 'video/mp4',
  '.mov': 'video/quicktime',
  '.webm': 'video/webm',
  '.mkv': 'video/x-matroska',
}

interface CachedMediaUpload {
  ref: BackendMediaRef
  expiresAtMs: number
}

const MEDIA_EXPIRY_SAFETY_WINDOW_MS = 5 * 60 * 1000
const MAX_MEDIA_CACHE_AGE_MS = 23 * 60 * 60 * 1000
const mediaCache = new Map<string, Promise<CachedMediaUpload>>()
const artifactCache = new Map<string, Promise<string>>()
let cleanedTransferDirectory = false

function requireExternalBackend(): { url: string; token: string } {
  if (getBackendConnectionMode() !== 'external') {
    throw new Error('Remote media transfer is only available for external backends')
  }
  const url = getBackendUrl()
  const token = getAuthToken()
  if (!url || !token) throw new Error('External backend is not connected')
  return { url, token }
}

function resolveMediaPath(filePath: string): string {
  if (!path.isAbsolute(filePath)) throw new Error('Media source path must be absolute')
  const resolvedPath = path.resolve(filePath)
  if (!fs.existsSync(resolvedPath) || !fs.statSync(resolvedPath).isFile()) {
    throw new Error(`Media source file does not exist: ${resolvedPath}`)
  }
  return resolvedPath
}

function requestModule(url: URL): typeof http | typeof https {
  return url.protocol === 'https:' ? https : http
}

function safeFilename(value: string): string {
  const base = path.basename(value).replace(/[^a-zA-Z0-9._-]/g, '_')
  return base && base !== '.' && base !== '..' ? base : 'artifact.bin'
}

function transferDirectory(): string {
  const directory = path.join(app.getPath('temp') || os.tmpdir(), 'ltx-desktop', 'backend-transfers')
  fs.mkdirSync(directory, { recursive: true })
  if (!cleanedTransferDirectory) {
    cleanedTransferDirectory = true
    const cutoff = Date.now() - 24 * 60 * 60 * 1000
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      if (!entry.isFile()) continue
      const entryPath = path.join(directory, entry.name)
      try {
        if (fs.statSync(entryPath).mtimeMs < cutoff) fs.unlinkSync(entryPath)
      } catch {
        // Best-effort cache cleanup.
      }
    }
  }
  return directory
}

async function readErrorResponse(response: http.IncomingMessage): Promise<string> {
  const chunks: Buffer[] = []
  for await (const chunk of response) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
    if (chunks.reduce((total, item) => total + item.length, 0) > 1024 * 1024) break
  }
  return Buffer.concat(chunks).toString('utf8').slice(0, 2000)
}

async function uploadFile(filePath: string, mediaType: BackendMediaType): Promise<BackendMediaRef> {
  const { url, token } = requireExternalBackend()
  const resolvedPath = resolveMediaPath(filePath)
  const stat = fs.statSync(resolvedPath)
  if (!stat.isFile()) throw new Error('Media source is not a file')

  const filename = safeFilename(resolvedPath)
  const contentType = CONTENT_TYPES[path.extname(filename).toLowerCase()] ?? 'application/octet-stream'
  const boundary = `----ltxdesktop-${crypto.randomBytes(18).toString('hex')}`
  const escapedFilename = filename.replace(/"/g, '_')
  const preamble = Buffer.from(
    `--${boundary}\r\nContent-Disposition: form-data; name="media_type"\r\n\r\n${mediaType}\r\n`
    + `--${boundary}\r\nContent-Disposition: form-data; name="file"; filename="${escapedFilename}"\r\n`
    + `Content-Type: ${contentType}\r\n\r\n`,
  )
  const suffix = Buffer.from(`\r\n--${boundary}--\r\n`)
  const endpoint = new URL('/api/media', url)

  return await new Promise<BackendMediaRef>((resolve, reject) => {
    const request = requestModule(endpoint).request(endpoint, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': `multipart/form-data; boundary=${boundary}`,
        'Content-Length': String(preamble.length + stat.size + suffix.length),
      },
    }, (response) => {
      void (async () => {
        if (!response.statusCode || response.statusCode < 200 || response.statusCode >= 300) {
          reject(new Error(`Media upload failed (HTTP ${response.statusCode ?? 0}): ${await readErrorResponse(response)}`))
          return
        }
        const chunks: Buffer[] = []
        for await (const chunk of response) {
          chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
        }
        try {
          resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')) as BackendMediaRef)
        } catch {
          reject(new Error('Media upload returned invalid JSON'))
        }
      })().catch(reject)
    })
    request.on('error', reject)
    request.setTimeout(30_000, () => request.destroy(new Error('Media upload timed out')))
    request.write(preamble)
    const source = fs.createReadStream(resolvedPath)
    source.on('error', reject)
    source.on('end', () => {
      request.end(suffix)
    })
    source.pipe(request, { end: false })
  })
}

export async function uploadBackendMedia(
  filePath: string,
  mediaType: BackendMediaType,
): Promise<BackendMediaRef> {
  const resolvedPath = resolveMediaPath(filePath)
  const stat = fs.statSync(resolvedPath)
  const { url, token } = requireExternalBackend()
  const connectionKey = crypto.createHash('sha256').update(`${url}\0${token}`).digest('hex')
  const key = `${connectionKey}:${mediaType}:${resolvedPath}:${stat.size}:${stat.mtimeMs}`
  const cached = mediaCache.get(key)
  if (cached) {
    const cachedUpload = await cached
    if (cachedUpload.expiresAtMs - MEDIA_EXPIRY_SAFETY_WINDOW_MS > Date.now()) {
      return cachedUpload.ref
    }
    mediaCache.delete(key)
  }

  const pending = uploadFile(resolvedPath, mediaType)
    .then((ref) => {
      const serverExpiresAtMs = Date.parse(ref.expires_at)
      if (!Number.isFinite(serverExpiresAtMs)) {
        throw new Error('Media upload returned an invalid expiry time')
      }
      // Bound the server timestamp by a local age so clock skew cannot make a
      // cache entry outlive the backend's 24-hour upload TTL.
      const expiresAtMs = Math.min(serverExpiresAtMs, Date.now() + MAX_MEDIA_CACHE_AGE_MS)
      return { ref, expiresAtMs }
    })
    .catch((error) => {
      mediaCache.delete(key)
      throw error
    })
  mediaCache.set(key, pending)
  return (await pending).ref
}

async function deleteArtifact(artifactId: string): Promise<void> {
  const { url, token } = requireExternalBackend()
  try {
    await fetch(`${url}/api/artifacts/${encodeURIComponent(artifactId)}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` },
    })
  } catch (error) {
    logger.warn(`Failed to delete downloaded backend artifact: ${error}`)
  }
}

async function downloadArtifact(artifact: BackendArtifactRef): Promise<string> {
  const { url, token } = requireExternalBackend()
  if (!artifact.artifact_id || artifact.artifact_id.length > 256) throw new Error('Invalid backend artifact ID')
  const filename = safeFilename(artifact.filename)
  const unique = crypto.randomBytes(8).toString('hex')
  const finalPath = path.join(transferDirectory(), `${unique}-${filename}`)
  const partialPath = `${finalPath}.part`
  const endpoint = new URL(`/api/artifacts/${encodeURIComponent(artifact.artifact_id)}`, url)

  try {
    await new Promise<void>((resolve, reject) => {
      const request = requestModule(endpoint).request(endpoint, {
        method: 'GET',
        headers: { Authorization: `Bearer ${token}` },
      }, (response) => {
        void (async () => {
          if (!response.statusCode || response.statusCode < 200 || response.statusCode >= 300) {
            reject(new Error(`Artifact download failed (HTTP ${response.statusCode ?? 0}): ${await readErrorResponse(response)}`))
            return
          }
          const hash = crypto.createHash('sha256')
          let receivedBytes = 0
          response.on('data', (chunk: Buffer) => {
            hash.update(chunk)
            receivedBytes += chunk.length
          })
          await pipeline(response, fs.createWriteStream(partialPath, { flags: 'wx' }))
          if (receivedBytes !== artifact.size_bytes) {
            throw new Error(`Artifact size mismatch: expected ${artifact.size_bytes}, received ${receivedBytes}`)
          }
          const digest = hash.digest('hex')
          if (artifact.sha256 && digest.toLowerCase() !== artifact.sha256.toLowerCase()) {
            throw new Error('Artifact checksum verification failed')
          }
          fs.renameSync(partialPath, finalPath)
          resolve()
        })().catch(reject)
      })
      request.on('error', reject)
      request.setTimeout(30_000, () => request.destroy(new Error('Artifact download timed out')))
      request.end()
    })
    await deleteArtifact(artifact.artifact_id)
    return finalPath
  } catch (error) {
    try { if (fs.existsSync(partialPath)) fs.unlinkSync(partialPath) } catch { /* best effort */ }
    throw error
  }
}

export async function materializeBackendArtifact(artifact: BackendArtifactRef): Promise<string> {
  const cached = artifactCache.get(artifact.artifact_id)
  if (cached) return cached
  const pending = downloadArtifact(artifact).catch((error) => {
    artifactCache.delete(artifact.artifact_id)
    throw error
  })
  artifactCache.set(artifact.artifact_id, pending)
  return pending
}
