import path from 'path'
import { fileURLToPath } from 'url'

const isWindows = process.platform === 'win32'
const MAX_TEMP_RECORDING_BYTES = 100 * 1024 * 1024

function normalize(p: string): string {
  return isWindows ? path.resolve(p).toLowerCase() : path.resolve(p)
}

const approvedPaths = new Set<string>()

export function approvePath(filePath: string): void {
  approvedPaths.add(normalize(filePath))
}

export function resolvePathWithinDirectory(directory: string, childPath: string): string {
  const root = path.resolve(directory)
  const resolved = path.resolve(root, childPath)
  const relative = path.relative(root, resolved)

  if (
    relative === '' ||
    relative.startsWith(`..${path.sep}`) ||
    path.isAbsolute(relative)
  ) {
    throw new Error('Resolved path must remain within the directory')
  }

  return resolved
}

export function validateTempRecordingInput(
  suffix: unknown,
  data: unknown,
): asserts data is ArrayBuffer {
  if (suffix !== '.wav') {
    throw new Error('Only WAV recording files are supported')
  }
  if (!(data instanceof ArrayBuffer)) {
    throw new Error('Recording data must be an ArrayBuffer')
  }
  if (data.byteLength > MAX_TEMP_RECORDING_BYTES) {
    throw new Error('Recording data exceeds the maximum supported size')
  }
}

export function validatePath(inputPath: string, allowedRoots: string[]): string {
  // fileURLToPath correctly handles the leading slash (POSIX file:///Users/... →
  // /Users/...), Windows drive letters (file:///C:/... → C:\...) and %-decoding.
  const cleaned = inputPath.startsWith('file://') ? fileURLToPath(inputPath) : inputPath
  const resolved = path.resolve(cleaned)
  const norm = normalize(resolved)

  for (const root of allowedRoots.map(normalize)) {
    if (norm === root || norm.startsWith(root + path.sep)) return resolved
  }

  let found = false
  approvedPaths.forEach((approved) => {
    if (norm === approved || norm.startsWith(approved + path.sep)) found = true
  })
  if (found) return resolved

  throw new Error(`Path not allowed: ${inputPath}`)
}
