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

  handle('getLogs', ({ query }) => {
    try {
      const logPath = getCurrentLogFilename()
      if (fs.existsSync(logPath)) {
        const content = fs.readFileSync(logPath, 'utf-8')
        const allLines = content.split('\n').map(l => l.trimEnd())
        // A search needs the whole session to find matches, not just the
        // recent tail we show by default -- skip the truncation while searching.
        const lines = query ? allLines : allLines.slice(-200)
        return { logPath, lines }
      }
      return { logPath, lines: [] }
    } catch (error) {
      logger.error(`Error getting logs: ${error}`)
      return { logPath: '', lines: [], error: String(error) }
    }
  })

  // Shows the active session log in the file manager, selected. Falls back to the folder
  // when the file is not there yet.
  handle('revealSessionLog', async () => {
    try {
      const logPath = getCurrentLogFilename()
      const logDir = getLogDir()
      const { shell } = await import('electron')
      if (fs.existsSync(logPath)) {
        shell.showItemInFolder(logPath)
        return true
      }
      if (fs.existsSync(logDir)) {
        await shell.openPath(logDir)
        return true
      }
      return false
    } catch (error) {
      logger.error(`Error revealing the session log: ${error}`)
      return false
    }
  })
}
