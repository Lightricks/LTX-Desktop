type LogLevel = 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG'

function log(level: LogLevel, consoleMethod: 'log' | 'warn' | 'error', message: string): void {
  console[consoleMethod](message)
  window.electronAPI?.writeLog?.({ level, message })?.catch(() => {})
}

export const logger = {
  info: (message: string) => log('INFO', 'log', message),
  warn: (message: string) => log('WARNING', 'warn', message),
  error: (message: string) => log('ERROR', 'error', message),
  debug: (message: string) => log('DEBUG', 'log', message),
}
