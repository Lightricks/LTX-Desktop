import path from 'path'
import fs from 'fs'

const isWindows = process.platform === 'win32'

function normalize(p: string): string {
  return isWindows ? path.resolve(p).toLowerCase() : path.resolve(p)
}

function stripFileUrl(fileUrl: string): string {
  let raw = fileUrl
  if (raw.startsWith('file:///')) raw = raw.slice(8)
  else if (raw.startsWith('file://')) raw = raw.slice(7)
  return decodeURIComponent(raw).replace(/\//g, path.sep)
}

const approvedPaths = new Set<string>()
const approvedFiles = new Set<string>()

export function approvePath(filePath: string): void {
  approvedPaths.add(normalize(filePath))
}

export function approveExistingFilePath(filePath: string): string {
  if (!path.isAbsolute(filePath)) {
    throw new Error(`Selected file path must be absolute: ${filePath}`)
  }

  const resolved = path.resolve(filePath)
  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) {
    throw new Error(`Selected file does not exist: ${resolved}`)
  }

  approvedFiles.add(normalize(resolved))
  return resolved
}

export function validatePath(inputPath: string, allowedRoots: string[]): string {
  const cleaned = inputPath.startsWith('file://') ? stripFileUrl(inputPath) : inputPath
  const resolved = path.resolve(cleaned)
  const norm = normalize(resolved)

  for (const root of allowedRoots.map(normalize)) {
    if (norm === root || norm.startsWith(root + path.sep)) return resolved
  }

  if (approvedFiles.has(norm)) return resolved

  let found = false
  approvedPaths.forEach((approved) => {
    if (norm === approved || norm.startsWith(approved + path.sep)) found = true
  })
  if (found) return resolved

  throw new Error(`Path not allowed: ${inputPath}`)
}
