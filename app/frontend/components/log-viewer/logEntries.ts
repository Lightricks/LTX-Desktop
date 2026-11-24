export type LogLevel = 'error' | 'warning' | 'info' | 'debug'

export const LOG_LEVELS: ReadonlyArray<{ id: LogLevel; label: string; hint?: string }> = [
  {
    id: 'error',
    label: 'Error',
    // electron/python-backend.ts logs every stderr line of the Python process as ERROR.
    // That stream carries tqdm progress bars and library warnings, not only failures.
    hint: 'Not every line here is a failure. The inference engine writes progress bars (such as denoising steps) and library warnings to its error stream, and the app records those lines as errors. Real failures, such as Python tracebacks, are listed here too.',
  },
  { id: 'warning', label: 'Warning' },
  { id: 'info', label: 'Info' },
  { id: 'debug', label: 'Debug' },
]

// Session log line written by electron/logger.ts: `<time> - <LEVEL> - [<Source>] <message>`.
// The level is read from the header only, so a message that contains ` - ERROR - ` keeps its level.
const LINE_HEADER = /^\d{4}-\d{2}-\d{2} [\d:,]+ - (CRITICAL|ERROR|WARNING|INFO|DEBUG) - /
const BACKEND_LINE = /^\d{4}-\d{2}-\d{2} [\d:,]+ - \w+ - \[Backend\] (.*)$/
// Python `logging` default format, as printed by the backend on stdout: `LEVEL:logger:message`.
const PYTHON_LOG_PREFIX = /^(CRITICAL|ERROR|WARNING|INFO|DEBUG):/
// Level names of the file header and of Python `logging`.
const LEVEL_BY_NAME: Record<string, LogLevel> = {
  CRITICAL: 'error',
  ERROR: 'error',
  WARNING: 'warning',
  INFO: 'info',
  DEBUG: 'debug',
}
const TRACEBACK_START = /^Traceback \(most recent call last\):/
// Sentences Python prints between the tracebacks of a chained exception.
const CHAIN_CONNECTOR =
  /^(The above exception was the direct cause of the following exception|During handling of the above exception, another exception occurred):/
// Stack frames, code lines and the "...<N lines>..." marker. The exception line is not indented.
const TRACEBACK_FRAME = /^(\s|\.\.\.<\d+ lines>)/
// A `File "...", line N` frame. It marks a traceback even when the opening line is not in view.
const TRACEBACK_FILE_FRAME = /^\s+File ".*", line \d+/

export interface LogEntry {
  text: string
  /** Level of this line. `null` for continuation lines of a multi-line message. */
  level: LogLevel | null
  /** Level used by the level filter. Continuation lines inherit the line above. */
  group: LogLevel | null
}

function detectLevel(line: string): LogLevel | null {
  const name = LINE_HEADER.exec(line)?.[1]
  return name ? LEVEL_BY_NAME[name] : null
}

/**
 * Levels for the backend, whose real severity is not the label Electron gives it.
 * Electron labels stdout lines INFO and stderr lines ERROR (see electron/python-backend.ts).
 * Python `WARNING:...` records and tracebacks arrive on stdout, so they look like INFO.
 */
export function parseLogEntries(lines: readonly string[]): LogEntry[] {
  let group: LogLevel | null = null
  let inTraceback = false
  return lines.map((text) => {
    let level = detectLevel(text)
    const message = BACKEND_LINE.exec(text)?.[1]
    if (message !== undefined) {
      const pythonLevel = PYTHON_LOG_PREFIX.exec(message)?.[1]
      if (pythonLevel) {
        level = LEVEL_BY_NAME[pythonLevel]
        inTraceback = false
      } else if (TRACEBACK_START.test(message)) {
        level = 'error'
        inTraceback = true
      } else if (CHAIN_CONNECTOR.test(message)) {
        level = 'error'
      } else if (inTraceback) {
        level = 'error'
        // The first line that is not a frame is the exception line, which ends the block.
        if (!TRACEBACK_FRAME.test(message)) inTraceback = false
      } else if (TRACEBACK_FILE_FRAME.test(message)) {
        // The opening line scrolled out of the 200-line tail. Treat the frame as the start.
        level = 'error'
        inTraceback = true
      }
    }
    if (level) group = level
    return { text, level, group }
  })
}

/** Keeps entries whose level is enabled. With every level enabled, nothing is hidden. */
export function filterByLevels(
  entries: LogEntry[],
  enabled: ReadonlySet<LogLevel>,
): LogEntry[] {
  if (enabled.size === LOG_LEVELS.length) return entries
  return entries.filter((entry) => entry.group !== null && enabled.has(entry.group))
}
