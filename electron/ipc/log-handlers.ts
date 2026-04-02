import fs from 'fs'
import { getLogDir, getCurrentLogFilename } from '../logging-management'
import { logger, writeLog } from '../logger'
import { handle } from './typed-handle'

const VALID_LOG_LEVELS = new Set(['INFO', 'WARNING', 'ERROR', 'DEBUG'])

export function registerLogHandlers(): void {
  handle('writeLog', ({ level, message }) => {
    const upperLevel = String(level).toUpperCase()
    if (!VALID_LOG_LEVELS.has(upperLevel)) return
    writeLog(upperLevel as 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG', 'Renderer', String(message))
  })

  handle('getLogs', () => {
    try {
      const logPath = getCurrentLogFilename()
      if (fs.existsSync(logPath)) {
        const content = fs.readFileSync(logPath, 'utf-8')
        const allLines = content.split('\n')
        const lines = allLines.slice(-200).map(l => l.trimEnd())
        return { logPath, lines }
      }
      return { logPath, lines: [] }
    } catch (error) {
      logger.error(`Error getting logs: ${error}`)
      return { logPath: '', lines: [], error: String(error) }
    }
  })

  handle('getLogPath', () => {
    const logPath = getCurrentLogFilename()
    const logDir = getLogDir()
    return { logPath, logDir }
  })

  handle('openLogFolder', async () => {
    const logDir = getLogDir()
    if (fs.existsSync(logDir)) {
      const { shell } = await import('electron')
      shell.openPath(logDir)
      return true
    }
    return false
  })
}
