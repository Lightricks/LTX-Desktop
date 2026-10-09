import { execSync } from 'child_process'
import fs from 'fs'
import path from 'path'
import { getCurrentDir, isDev } from './config'
import { logger, setLogFilePath } from './logger'
export { getLogDir } from './app-paths'
import { getLogDir } from './app-paths'

let currentLogFilename: string | null = null

function getGitCommitHash(): string {
  try {
    const cwd = isDev ? getCurrentDir() : process.resourcesPath
    return execSync('git rev-parse --short HEAD', { cwd, encoding: 'utf-8', stdio: ['ignore', 'pipe', 'ignore'] }).trim()
  } catch {
    return 'unknown'
  }
}

function generateLogFilename(gitHash: string): string {
  const now = new Date()
  const ts = now.toISOString().replace(/T/, '_').replace(/:/g, '-').replace(/\..+/, '')
  return `session_${ts}_${gitHash}.log`
}

function cleanupOldLogs(logDir: string, maxFiles = 100): void {
  try {
    const files = fs.readdirSync(logDir)
      .filter(f => (f.startsWith('session_') || f.startsWith('backend_')) && f.endsWith('.log'))
      .map(f => ({ name: f, time: fs.statSync(path.join(logDir, f)).mtimeMs }))
      .sort((a, b) => a.time - b.time)

    const toDelete = files.slice(0, Math.max(0, files.length - maxFiles))
    for (const f of toDelete) {
      try {
        fs.unlinkSync(path.join(logDir, f.name))
      } catch {
        // ignore deletion errors
      }
    }
  } catch {
    // ignore cleanup errors
  }
}

export function initSessionLog(): void {
  const logDir = getLogDir()
  fs.mkdirSync(logDir, { recursive: true })

  const gitHash = getGitCommitHash()
  currentLogFilename = path.join(logDir, generateLogFilename(gitHash))

  cleanupOldLogs(logDir)

  setLogFilePath(currentLogFilename)
  logger.info(`Session log file: ${currentLogFilename}`)
}

export function getCurrentLogFilename(): string {
  if (!currentLogFilename) {
    throw new Error('initSessionLog() must be called before getCurrentLogFilename()')
  }
  return currentLogFilename
}
