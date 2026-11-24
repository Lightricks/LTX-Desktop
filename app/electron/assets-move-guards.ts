import { isFfmpegRunning } from './export/ffmpeg-utils'
import { getAuthToken, getBackendUrl, isGenerationActive } from './python-backend'

/** Folders the user must not pick: the move would damage the system. */
export function systemFolders(): string[] {
  if (process.platform === 'win32') {
    return [process.env.SystemRoot, process.env.ProgramFiles, process.env['ProgramFiles(x86)'], process.env.ProgramData]
      .filter((dir): dir is string => Boolean(dir))
  }
  if (process.platform === 'darwin') return ['/System', '/usr', '/bin', '/sbin', '/etc', '/private/etc', '/Library']
  return ['/usr', '/bin', '/sbin', '/etc', '/lib', '/boot', '/proc', '/sys', '/dev']
}

/**
 * Why the app server cannot restart now, or `null`. The backend queue is the source of truth.
 * A queued generation also needs the backend, and an ffmpeg job reads the asset files.
 */
export async function busyReason(): Promise<string | null> {
  if (isGenerationActive()) return 'A generation is running.'
  if (isFfmpegRunning()) return 'An export is running.'
  const url = getBackendUrl()
  if (!url) return null
  try {
    const response = await fetch(`${url}/api/generation-queue`, {
      headers: { Authorization: `Bearer ${getAuthToken() ?? ''}` },
      signal: AbortSignal.timeout(3000),
    })
    if (!response.ok) return 'The app cannot check for running generations.'
    const snapshot = (await response.json()) as { active?: unknown[]; queued?: unknown[] }
    const count = (snapshot.active?.length ?? 0) + (snapshot.queued?.length ?? 0)
    return count > 0 ? 'A generation is running or queued.' : null
  } catch {
    return 'The app cannot check for running generations.'
  }
}
