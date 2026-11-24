import { useRef, type KeyboardEvent, type ReactNode } from 'react'
import { clsx } from 'clsx'
import { Check, ChevronDown, ChevronUp, Copy, X } from 'lucide-react'
import { Button } from '@ds/Button/Button'
import { Text } from '@ds/Text/Text'
import { TextField } from '@ds/TextField/TextField'
import { LogActionsMenu } from './LogActionsMenu'
import { LogTooltip } from './LogTooltip'
import { LogFontSizePicker } from './LogFontSizePicker'
import { LOG_LEVELS, type LogLevel } from './logEntries'
import type { LogFontSizeId } from './useLogViewPrefs'
import styles from './LogViewer.module.scss'

function ToolbarIconButton({
  label,
  icon,
  pressed,
  disabled,
  onClick,
}: {
  label: string
  icon: ReactNode
  pressed?: boolean
  disabled?: boolean
  onClick: () => void
}) {
  return (
    <LogTooltip content={label}>
      <Button
        hierarchy="plain"
        size="md"
        isIconOnly
        aria-label={label}
        aria-pressed={pressed}
        leftIcon={icon}
        disabled={disabled}
        onClick={onClick}
      />
    </LogTooltip>
  )
}

function LevelDot({ level }: { level: LogLevel }) {
  return (
    <svg viewBox="0 0 8 8" aria-hidden className={styles[`dot_${level}`]}>
      <circle cx="4" cy="4" r="3.5" fill="currentColor" />
    </svg>
  )
}

export interface LogToolbarProps {
  search: {
    input: string
    onInputChange: (value: string) => void
    onClear: () => void
    matchCase: boolean
    onMatchCaseChange: (value: boolean) => void
    useRegex: boolean
    onUseRegexChange: (value: boolean) => void
    isInvalid: boolean
    isActive: boolean
    matchCount: number
    currentMatchNumber: number
    onNext: () => void
    onPrevious: () => void
  }
  levels: {
    enabled: ReadonlySet<LogLevel>
    onToggle: (level: LogLevel) => void
  }
  view: {
    fontSize: LogFontSizeId
    onFontSizeChange: (value: string) => void
    wrap: boolean
    onWrapChange: (value: boolean) => void
  }
  actions: {
    isLoading: boolean
    hasLines: boolean
    copied: boolean
    onRefresh: () => void
    onCopy: () => void
    onDownload: () => void
    onRevealLog: () => void
  }
}

/** The text for the match counter. Empty when there is nothing to show. */
function matchSummary(search: LogToolbarProps['search']): string {
  if (!search.isActive && !search.isInvalid) return ''
  if (search.isInvalid) return 'Invalid regex'
  if (search.matchCount === 0) return 'No matches'
  // A line counts once, however many matches it holds.
  return `${search.currentMatchNumber} of ${search.matchCount} lines`
}

/** Row 1 finds text. Row 2 filters, then holds the font size, copy and the actions menu. */
export function LogToolbar({ search, levels, view, actions }: LogToolbarProps) {
  // Enter in the search field jumps to the next match, Shift+Enter to the previous one.
  // The toggles share the wrapper, so ignore Enter on them: it must still press the button.
  const handleSearchKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'Enter' || !(event.target instanceof HTMLInputElement)) return
    event.preventDefault()
    if (event.shiftKey) search.onPrevious()
    else search.onNext()
  }
  // The clear button unmounts, so move focus back to the field to keep typing and keyboard use.
  const searchInputRef = useRef<HTMLInputElement>(null)
  const handleClear = () => {
    search.onClear()
    searchInputRef.current?.focus()
  }
  const hasMatches = search.matchCount > 0
  const summary = matchSummary(search)

  return (
    <div className={styles.toolbar}>
      <div className={styles.toolbarRow}>
        <div className={styles.searchBox} onKeyDown={handleSearchKeyDown}>
          <TextField
            ref={searchInputRef}
            className={styles.searchInput}
            value={search.input}
            onInputChange={search.onInputChange}
            aria-label="Search logs"
            placeholder="Search logs..."
          />
          <div className={styles.searchToggles}>
            {search.input && (
              <LogTooltip content="Clear search">
                <Button
                  hierarchy="plain"
                  size="xs"
                  isIconOnly
                  aria-label="Clear search"
                  leftIcon={<X aria-hidden />}
                  onClick={handleClear}
                />
              </LogTooltip>
            )}
            <LogTooltip content="Match case">
              <Button
                hierarchy="plain"
                size="xs"
                label="Aa"
                aria-label="Match case"
                aria-pressed={search.matchCase}
                onClick={() => search.onMatchCaseChange(!search.matchCase)}
              />
            </LogTooltip>
            <LogTooltip content="Use regular expression">
              <Button
                hierarchy="plain"
                size="xs"
                label=".*"
                aria-label="Use regular expression"
                aria-pressed={search.useRegex}
                onClick={() => search.onUseRegexChange(!search.useRegex)}
              />
            </LogTooltip>
          </div>
        </div>
        {/* Always mounted, so a screen reader announces "No matches" and "Invalid regex".
            The visible text below is hidden from it to avoid a second reading. */}
        <span role="status" className="sr-only">{summary}</span>
        {summary && (
          <div aria-hidden="true" className={styles.matchCountWrap}>
            <Text
              variant="body"
              size="xs"
              className={clsx(styles.matchCount, search.isInvalid && styles.matchCountInvalid)}
            >
              {summary}
            </Text>
          </div>
        )}
        <div className={styles.group}>
          <ToolbarIconButton
            label="Previous match (Shift+Enter)"
            icon={<ChevronUp aria-hidden />}
            disabled={!hasMatches}
            onClick={search.onPrevious}
          />
          <ToolbarIconButton
            label="Next match (Enter)"
            icon={<ChevronDown aria-hidden />}
            disabled={!hasMatches}
            onClick={search.onNext}
          />
        </div>
      </div>

      <div className={styles.toolbarRow}>
        <div className={styles.group} role="group" aria-label="Log levels">
          {LOG_LEVELS.map((level) => (
            <LogTooltip key={level.id} content={level.hint} maxWidth={260} disabled={!level.hint}>
              <Button
                hierarchy="plain"
                size="md"
                label={level.label}
                leftIcon={<LevelDot level={level.id} />}
                aria-pressed={levels.enabled.has(level.id)}
                onClick={() => levels.onToggle(level.id)}
              />
            </LogTooltip>
          ))}
        </div>
        <div className={styles.spacer} />
        <div className={styles.group}>
          <LogFontSizePicker value={view.fontSize} onValueChange={view.onFontSizeChange} />
          <ToolbarIconButton
            label={actions.copied ? 'Copied' : 'Copy logs'}
            icon={actions.copied ? <Check aria-hidden /> : <Copy aria-hidden />}
            disabled={!actions.hasLines}
            onClick={actions.onCopy}
          />
          <LogActionsMenu
            wrap={view.wrap}
            onWrapChange={view.onWrapChange}
            isLoading={actions.isLoading}
            hasLines={actions.hasLines}
            onRefresh={actions.onRefresh}
            onDownload={actions.onDownload}
            onRevealLog={actions.onRevealLog}
          />
        </div>
      </div>
    </div>
  )
}
