import { useState, useEffect, useRef } from 'react'
import { X, FolderOpen, RefreshCw, ChevronDown, ChevronUp, Download, Search } from 'lucide-react'
import { Button } from './ui/button'
import { logger } from '../lib/logger'
import { getBackendCredentials } from '../lib/backend'

interface LogViewerProps {
  isOpen: boolean
  onClose: () => void
  embedded?: boolean
}

// Wraps every case-insensitive occurrence of `query` in the line with a <mark>,
// keeping all other text as-is -- marks matches instead of hiding non-matching
// lines, so surrounding context stays visible while searching.
function highlightMatches(line: string, query: string): React.ReactNode {
  if (!query) return line
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const parts = line.split(new RegExp(`(${escaped})`, 'gi'))
  if (parts.length === 1) return line
  // String.split with a capturing group alternates [text, match, text, match, ...] --
  // odd indices are always the captured matches.
  return parts.map((part, i) =>
    i % 2 === 1
      ? <mark key={i} className="bg-yellow-500/70 text-black rounded-sm px-0.5">{part}</mark>
      : <span key={i}>{part}</span>,
  )
}

export function LogViewer({ isOpen, onClose, embedded = false }: LogViewerProps) {
  const [logs, setLogs] = useState<string[]>([])
  const [logPath, setLogPath] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [autoScroll, setAutoScroll] = useState(true)
  const [searchInput, setSearchInput] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [tokenCopied, setTokenCopied] = useState(false)
  const logContainerRef = useRef<HTMLDivElement>(null)
  // Autoscroll doesn't make sense while browsing search matches -- remember
  // the user's prior preference so clearing the search restores it exactly.
  const savedAutoScrollRef = useRef<boolean | null>(null)

  // Debounce the raw input so typing doesn't fire an IPC round-trip (a full
  // untruncated log read while searching) on every keystroke.
  useEffect(() => {
    const timer = setTimeout(() => setSearchQuery(searchInput.trim()), 300)
    return () => clearTimeout(timer)
  }, [searchInput])

  const fetchLogs = async () => {
    if (!window.electronAPI?.getLogs) return

    setIsLoading(true)
    try {
      const result = await window.electronAPI.getLogs({ query: searchQuery || undefined })
      setLogs(result.lines || [])
      setLogPath(result.logPath || '')
    } catch (error) {
      logger.error(`Failed to fetch logs: ${error}`)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (!isOpen) return
    fetchLogs()
    // Live-tail only when NOT searching. A search reads the whole session file and
    // ships every line over IPC (then re-highlights each), so polling that every 2s is a
    // large CPU/IPC cost for no gain -- results are a point-in-time view. Fetch once per
    // query change instead (this effect re-runs on searchQuery), and skip the interval.
    if (searchQuery) return
    const interval = setInterval(fetchLogs, 2000)
    return () => clearInterval(interval)
  }, [isOpen, searchQuery])

  const matchCount = searchQuery
    ? logs.filter(l => l.toLowerCase().includes(searchQuery.toLowerCase())).length
    : 0

  useEffect(() => {
    const isSearching = Boolean(searchQuery)
    if (isSearching && savedAutoScrollRef.current === null) {
      savedAutoScrollRef.current = autoScroll
      setAutoScroll(false)
    } else if (!isSearching && savedAutoScrollRef.current !== null) {
      setAutoScroll(savedAutoScrollRef.current)
      savedAutoScrollRef.current = null
    }
    // autoScroll intentionally excluded -- this only reacts to searchQuery
    // transitions, not to the manual toggle (which is disabled while searching anyway).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchQuery])

  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  const handleOpenFolder = async () => {
    if (window.electronAPI?.openLogFolder) {
      await window.electronAPI.openLogFolder()
    }
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
      setTimeout(() => setTokenCopied(false), 1200)
    } catch (error) {
      logger.error(`Failed to copy backend token: ${error}`)
    }
  }

  const handleDownload = () => {
    const content = logs.join('\n')
    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = logPath ? logPath.split(/[/\\]/).pop() || 'session.log' : 'session.log'
    a.click()
    URL.revokeObjectURL(url)
  }

  if (!isOpen) return null

  const panel = (
    <div className={`bg-zinc-900 rounded-lg border border-zinc-700 w-full ${embedded ? 'h-full' : 'max-w-5xl h-[80vh]'} flex flex-col`}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-zinc-700">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-white">Logs</h2>
            <span className="text-xs text-zinc-500 font-mono truncate max-w-[300px]" title={logPath}>
              {logPath}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleDownload}
              disabled={logs.length === 0}
              className="text-zinc-400 hover:text-white"
              title="Download logs"
            >
              <Download className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleOpenFolder}
              className="text-zinc-400 hover:text-white"
              title="Open log folder"
            >
              <FolderOpen className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={fetchLogs}
              disabled={isLoading}
              className="text-zinc-400 hover:text-white"
              title="Refresh logs"
            >
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setAutoScroll(!autoScroll)}
              disabled={Boolean(searchQuery)}
              className={`${autoScroll ? 'text-blue-400' : 'text-zinc-400'} hover:text-white disabled:opacity-40 disabled:cursor-not-allowed`}
              title={searchQuery ? 'Auto-scroll disabled while searching' : autoScroll ? 'Auto-scroll enabled' : 'Auto-scroll disabled'}
            >
              {autoScroll ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
            </Button>
            {!embedded && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="text-zinc-400 hover:text-white"
              >
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>

        {/* Search */}
        <div className="flex items-center gap-2 px-4 py-2 border-b border-zinc-800">
          <Search className="h-3.5 w-3.5 text-zinc-500 flex-shrink-0" />
          <input
            type="text"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            placeholder="Search logs (case-insensitive)..."
            className="flex-1 bg-transparent text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none"
          />
          {searchQuery && (
            <span className="text-xs text-zinc-500 flex-shrink-0">
              {matchCount} matching line{matchCount === 1 ? '' : 's'}
            </span>
          )}
        </div>

        {/* Log content */}
        <div
          ref={logContainerRef}
          className="flex-1 overflow-auto p-4 font-mono text-xs bg-black"
        >
          {logs.length === 0 ? (
            <div className="text-zinc-500 text-center py-8">
              No logs yet...
            </div>
          ) : (
            <div className="space-y-0.5">
              {logs.map((line, index) => {
                // Color code log levels
                let lineClass = 'text-zinc-300'
                if (line.includes(' - ERROR - ') || line.includes(' - CRITICAL - ')) {
                  lineClass = 'text-red-400'
                } else if (line.includes(' - WARNING - ')) {
                  lineClass = 'text-yellow-400'
                } else if (line.includes(' - INFO - ')) {
                  lineClass = 'text-blue-300'
                } else if (line.includes(' - DEBUG - ')) {
                  lineClass = 'text-zinc-500'
                }
                
                return (
                  <div key={index} className={`${lineClass} whitespace-pre-wrap break-all`}>
                    {highlightMatches(line, searchQuery)}
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-3 border-t border-zinc-700 text-xs text-zinc-500">
          <span>{logs.length} lines {searchQuery ? '(full session)' : '(last 200)'}</span>
          <div className="flex items-center gap-3">
            {/* Semi-hidden: copies the backend Bearer token for local QA/perf scripts
                (the Electron-spawned backend's token is random per launch). Muted until
                hover so it stays out of the way. */}
            <button
              type="button"
              onClick={() => void handleCopyToken()}
              className="font-mono tracking-widest text-zinc-700 hover:text-zinc-300 transition-colors"
              title="Copy backend auth token"
              aria-label="Copy backend auth token"
            >
              {tokenCopied ? 'copied!' : '•••••'}
            </button>
            <span>Auto-refreshing every 2s</span>
          </div>
        </div>
      </div>
  )

  if (embedded) {
    return panel
  }

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
      {panel}
    </div>
  )
}
