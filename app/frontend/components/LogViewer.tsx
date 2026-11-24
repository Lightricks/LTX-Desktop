import { useMemo, useState } from 'react'
import { X } from 'lucide-react'
import { Text } from '@ds/Text/Text'
import { logger } from '../lib/logger'
import { getBackendCredentials } from '../lib/backend'
import { copyTextToClipboard } from '../lib/copy-to-clipboard'
import { AppProductDialogShell, ProductDialogIconButton } from './dialog/AppProductDialogShell'
import { LogPane } from './log-viewer/LogPane'
import { LogToolbar } from './log-viewer/LogToolbar'
import { filterByLevels, parseLogEntries } from './log-viewer/logEntries'
import { useCloseOnEscape } from './log-viewer/useCloseOnEscape'
import { useLevelFilter } from './log-viewer/useLevelFilter'
import { useSearchInput, useSearchMatches } from './log-viewer/useLogSearch'
import { useLogViewPrefs } from './log-viewer/useLogViewPrefs'
import { useSessionLogs } from './log-viewer/useSessionLogs'
import styles from './log-viewer/LogViewer.module.scss'

interface LogViewerProps {
  isOpen: boolean
  onClose: () => void
  embedded?: boolean
}

const DIALOG_TITLE_ID = 'log-viewer-dialog-title'
const COPIED_FEEDBACK_MS = 1200

export function LogViewer({ isOpen, onClose, embedded = false }: LogViewerProps) {
  const searchInput = useSearchInput()
  const { search } = searchInput
  // A valid search needs the whole session, so it also switches off the live tail.
  const isSearching = search.kind === 'ok'
  // False while the user has scrolled up to read. The live tail then pauses so the text stays put.
  const [isFollowing, setIsFollowing] = useState(true)
  const { logs, logPath, isLoading, refresh } = useSessionLogs(
    isOpen,
    search.kind === 'ok' ? search.query : '',
    isFollowing,
  )
  const levelFilter = useLevelFilter()
  const prefs = useLogViewPrefs()

  const allEntries = useMemo(() => parseLogEntries(logs), [logs])
  const entries = useMemo(
    () => filterByLevels(allEntries, levelFilter.enabled),
    [allEntries, levelFilter.enabled],
  )
  const matches = useSearchMatches(entries, search)

  // The embedded crash screen has nothing to close, so Escape does nothing there.
  useCloseOnEscape(onClose, isOpen && !embedded)

  const [logsCopied, setLogsCopied] = useState(false)
  const [tokenCopied, setTokenCopied] = useState(false)

  const handleRevealLog = async () => {
    if (window.electronAPI?.revealSessionLog) {
      await window.electronAPI.revealSessionLog()
    }
  }

  // Copy and download use the lines on screen, so a level filter applies to both.
  const handleCopyLogs = async () => {
    const copied = await copyTextToClipboard(entries.map((entry) => entry.text).join('\n'))
    if (!copied) {
      logger.error('Failed to copy logs')
      return
    }
    setLogsCopied(true)
    setTimeout(() => setLogsCopied(false), COPIED_FEEDBACK_MS)
  }

  // Copy the backend Bearer token so local QA/perf scripts can authenticate
  // against the Electron-spawned backend (its token is random per launch).
  // Surfaced via a semi-hidden control in the footer (see below).
  const handleCopyToken = async () => {
    try {
      const { token } = await getBackendCredentials()
      if (!token) {
        logger.error('No backend token available to copy')
        return
      }
      await navigator.clipboard.writeText(token)
      setTokenCopied(true)
      setTimeout(() => setTokenCopied(false), COPIED_FEEDBACK_MS)
    } catch (error) {
      logger.error(`Failed to copy backend token: ${error}`)
    }
  }

  const handleDownload = () => {
    const content = entries.map((entry) => entry.text).join('\n')
    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = logPath ? logPath.split(/[/\\]/).pop() || 'session.log' : 'session.log'
    a.click()
    URL.revokeObjectURL(url)
  }

  if (!isOpen) return null

  // The status says what the view does. A count shows only when it is not the plain tail.
  const footerStatus = [
    isSearching
      ? `Searching the full session (${logs.length} lines)`
      : isFollowing
        ? 'Auto-refreshing every 2s'
        : 'Paused while scrolled up',
    levelFilter.isFiltered ? `${entries.length} of ${logs.length} lines` : null,
  ]
    .filter(Boolean)
    .join(' · ')

  const panel = (
    <div className={styles.panel}>
      <LogToolbar
        search={{
          input: searchInput.input,
          onInputChange: searchInput.setInput,
          onClear: searchInput.clear,
          matchCase: searchInput.matchCase,
          onMatchCaseChange: searchInput.setMatchCase,
          useRegex: searchInput.useRegex,
          onUseRegexChange: searchInput.setUseRegex,
          isInvalid: search.kind === 'invalid',
          isActive: isSearching,
          matchCount: matches.matchCount,
          currentMatchNumber: matches.currentMatchNumber,
          onNext: matches.next,
          onPrevious: matches.previous,
        }}
        levels={{ enabled: levelFilter.enabled, onToggle: levelFilter.toggle }}
        view={{
          fontSize: prefs.fontSize,
          onFontSizeChange: prefs.setFontSize,
          wrap: prefs.wrap,
          onWrapChange: prefs.setWrap,
        }}
        actions={{
          isLoading,
          hasLines: entries.length > 0,
          copied: logsCopied,
          onRefresh: () => void refresh(),
          onCopy: () => void handleCopyLogs(),
          onDownload: handleDownload,
          onRevealLog: () => void handleRevealLog(),
        }}
      />

      {logPath && (
        <Text variant="body" size="xs" shouldTruncate title={logPath} className={styles.path}>
          {logPath}
        </Text>
      )}

      <LogPane
        entries={entries}
        lineRanges={matches.lineRanges}
        currentLine={matches.currentLine}
        scrollNonce={matches.scrollNonce}
        emptyLabel={logs.length === 0 ? 'No logs yet...' : 'No lines match the selected levels.'}
        fontSizePx={prefs.fontSizePx}
        wrap={prefs.wrap}
        followEnabled={!isSearching}
        onFollowingChange={setIsFollowing}
      />

      <div className={styles.footer}>
        <Text variant="body" size="xs">{footerStatus}</Text>
        {/* Semi-hidden: copies the backend Bearer token for local QA/perf scripts
            (the Electron-spawned backend's token is random per launch). */}
        <button
          type="button"
          onClick={() => void handleCopyToken()}
          className={styles.tokenButton}
          title="Copy backend auth token"
          aria-label="Copy backend auth token"
        >
          {tokenCopied ? 'copied!' : '•••••'}
        </button>
      </div>
    </div>
  )

  if (embedded) {
    return (
      <div className={styles.embedded}>
        <Text as="h3" variant="heading" size="sm">Diagnostic logs</Text>
        {panel}
      </div>
    )
  }

  return (
    <AppProductDialogShell
      aria-labelledby={DIALOG_TITLE_ID}
      title="Diagnostic logs"
      onBackdropClick={onClose}
      className="h-[80vh] max-w-5xl"
      headerActions={
        <ProductDialogIconButton aria-label="Close diagnostic logs" onClick={onClose}>
          <X className="h-4 w-4" />
        </ProductDialogIconButton>
      }
    >
      {panel}
    </AppProductDialogShell>
  )
}
