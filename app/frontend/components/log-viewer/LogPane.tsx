import { useEffect, type CSSProperties } from 'react'
import { clsx } from 'clsx'
import { ArrowDown } from 'lucide-react'
import { Button } from '@ds/Button/Button'
import { LogLines } from './LogLines'
import type { LogEntry } from './logEntries'
import type { MatchRange } from './logSearch'
import { useStickToBottom } from './useStickToBottom'
import styles from './LogViewer.module.scss'

interface LogPaneProps {
  entries: LogEntry[]
  lineRanges: MatchRange[][]
  currentLine: number
  scrollNonce: number
  emptyLabel: string
  fontSizePx: number
  wrap: boolean
  /** Follow new lines at the bottom. Off while the user browses search results. */
  followEnabled: boolean
  /** Called when the view sticks to the bottom (true) or the user scrolls away (false). */
  onFollowingChange: (isFollowing: boolean) => void
}

export function LogPane({
  entries,
  lineRanges,
  currentLine,
  scrollNonce,
  emptyLabel,
  fontSizePx,
  wrap,
  followEnabled,
  onFollowingChange,
}: LogPaneProps) {
  const { containerRef, contentRef, onScroll, scrollToBottom, showJumpButton, isStuck } =
    useStickToBottom(followEnabled)

  // Also runs on mount, so a reopened viewer starts as following.
  useEffect(() => onFollowingChange(isStuck), [isStuck, onFollowingChange])

  return (
    <div className={styles.paneFrame}>
      <div
        ref={containerRef}
        className={styles.logPane}
        style={{ '--log-font-size': `${fontSizePx}px` } as CSSProperties}
        onScroll={onScroll}
      >
        <div ref={contentRef} className={clsx(styles.content, !wrap && styles.noWrap)}>
          <LogLines
            entries={entries}
            lineRanges={lineRanges}
            currentLine={currentLine}
            scrollNonce={scrollNonce}
            emptyLabel={emptyLabel}
          />
        </div>
      </div>
      {showJumpButton && (
        <Button
          appearance="brand"
          hierarchy="primary"
          size="md"
          label="Jump to latest"
          leftIcon={<ArrowDown aria-hidden />}
          className={styles.jumpButton}
          onClick={scrollToBottom}
        />
      )}
    </div>
  )
}
