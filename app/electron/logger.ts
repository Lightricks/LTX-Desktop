import fs from 'fs'

type LogLevel = 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG'
type LogSource = 'Electron' | 'Renderer' | 'Backend'

let logFilePath: string | null = null

/** Called by initSessionLog() to tell the writer where to append. */
export function setLogFilePath(filePath: string): void {
  logFilePath = filePath
}

function formatTimestamp(): string {
  const now = new Date()
  const pad = (n: number, len = 2) => String(n).padStart(len, '0')
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())},${pad(now.getMilliseconds(), 3)}`
}

export function writeLog(level: LogLevel, source: LogSource, message: string): void {
  if (!logFilePath) {
    // Log file not yet initialized (early startup)
    return
  }

  const line = `${formatTimestamp()} - ${level} - [${source}] ${message}\n`
  try {
    fs.appendFileSync(logFilePath, line, 'utf-8')
  } catch {
    // Silently ignore write errors
  }
}

function log(level: LogLevel, consoleMethod: 'log' | 'warn' | 'error', message: string): void {
  console[consoleMethod](message)
  writeLog(level, 'Electron', message)
}

export const logger = {
  info: (message: string) => log('INFO', 'log', message),
  warn: (message: string) => log('WARNING', 'warn', message),
  error: (message: string) => log('ERROR', 'error', message),
  debug: (message: string) => log('DEBUG', 'log', message),
}
