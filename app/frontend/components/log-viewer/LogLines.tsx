import { memo, useEffect, useRef, type ReactNode } from 'react'
import { clsx } from 'clsx'
import { Text } from '@ds/Text/Text'
import type { LogEntry } from './logEntries'
import type { MatchRange } from './logSearch'
import styles from './LogViewer.module.scss'

// Marks every match range in the line. Other text stays as-is, so surrounding
// context stays visible while searching.
function highlightMatches(text: string, ranges: readonly MatchRange[] | undefined): ReactNode {
  if (!ranges || ranges.length === 0) return text
  const parts: ReactNode[] = []
  let cursor = 0
  ranges.forEach(([start, end], i) => {
    if (start > cursor) parts.push(text.slice(cursor, start))
    parts.push(<mark key={i} className={styles.match}>{text.slice(start, end)}</mark>)
    cursor = end
  })
  if (cursor < text.length) parts.push(text.slice(cursor))
  return parts
}

interface LogLineProps {
  entry: LogEntry
  ranges: readonly MatchRange[] | undefined
  isCurrent: boolean
  /** Changes on each next/previous action. Only the current line gets it, so no other row re-renders. */
  scrollNonce: number
}

// Memoized: stepping through matches re-renders only the two rows whose props change.
const LogLine = memo(function LogLine({ entry, ranges, isCurrent, scrollNonce }: LogLineProps) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isCurrent) ref.current?.scrollIntoView({ block: 'center', inline: 'nearest' })
  }, [isCurrent, scrollNonce])

  return (
    <div
      ref={ref}
      className={clsx(styles.line, entry.level && styles[entry.level], isCurrent && styles.currentMatch)}
    >
      {highlightMatches(entry.text, ranges)}
    </div>
  )
})

interface LogLinesProps {
  entries: LogEntry[]
  /** Match ranges per entry, in the same order as `entries`. Empty when not searching. */
  lineRanges: MatchRange[][]
  /** Index in `entries` of the current search match, or -1. */
  currentLine: number
  /** Changes on each next/previous action, so the current match scrolls into view again. */
  scrollNonce: number
  emptyLabel: string
}

export function LogLines({ entries, lineRanges, currentLine, scrollNonce, emptyLabel }: LogLinesProps) {
  if (entries.length === 0) {
    return (
      <Text variant="body" size="sm" align="center" className={styles.empty}>
        {emptyLabel}
      </Text>
    )
  }

  return (
    <div className={styles.lines}>
      {entries.map((entry, index) => {
        const isCurrent = index === currentLine
        return (
          <LogLine
            key={index}
            entry={entry}
            ranges={lineRanges[index]}
            isCurrent={isCurrent}
            scrollNonce={isCurrent ? scrollNonce : 0}
          />
        )
      })}
    </div>
  )
}
