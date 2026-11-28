import { useState } from 'react'
import { LOG_LEVELS, type LogLevel } from './logEntries'

/** Which log levels are visible. Every level starts enabled. */
export function useLevelFilter() {
  const [enabled, setEnabled] = useState<ReadonlySet<LogLevel>>(
    () => new Set(LOG_LEVELS.map((level) => level.id)),
  )

  const toggle = (level: LogLevel) =>
    setEnabled((current) => {
      const next = new Set(current)
      if (next.has(level)) next.delete(level)
      else next.add(level)
      return next
    })

  return { enabled, toggle, isFiltered: enabled.size < LOG_LEVELS.length }
}
