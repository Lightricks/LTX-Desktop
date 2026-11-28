import { useCallback, useEffect, useState } from 'react'
import { logger } from '../../lib/logger'

const LIVE_TAIL_INTERVAL_MS = 2000

/**
 * Fetches the session log and live-tails it.
 * A non-empty `searchQuery` asks for the full session instead of the recent tail.
 * The live tail runs only while `isFollowing` is true. A new tail replaces the old
 * lines, so while the user reads further up, an update would move the text.
 */
export function useSessionLogs(isOpen: boolean, searchQuery: string, isFollowing: boolean) {
  const [logs, setLogs] = useState<string[]>([])
  const [logPath, setLogPath] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  // `showLoading` is true for fetches the user can see (open, new search, Refresh).
  // A background poll does not set it, so the Refresh button does not flicker.
  const fetchLogs = useCallback(
    async (showLoading: boolean) => {
      if (!window.electronAPI?.getLogs) return

      if (showLoading) setIsLoading(true)
      try {
        const result = await window.electronAPI.getLogs({ query: searchQuery || undefined })
        setLogs(result.lines || [])
        setLogPath(result.logPath || '')
      } catch (error) {
        logger.error(`Failed to fetch logs: ${error}`)
      } finally {
        if (showLoading) setIsLoading(false)
      }
    },
    [searchQuery],
  )

  const refresh = useCallback(() => fetchLogs(true), [fetchLogs])

  // Fetch once on open and once per query change.
  useEffect(() => {
    if (isOpen) void fetchLogs(true)
  }, [isOpen, fetchLogs])

  // Live-tail only when NOT searching. A search reads the whole session file and
  // ships every line over IPC (then re-highlights each), so polling that every 2s is a
  // large CPU/IPC cost for no gain -- results are a point-in-time view.
  useEffect(() => {
    if (!isOpen || searchQuery || !isFollowing) return
    const interval = setInterval(() => void fetchLogs(false), LIVE_TAIL_INTERVAL_MS)
    return () => clearInterval(interval)
  }, [isOpen, searchQuery, isFollowing, fetchLogs])

  return { logs, logPath, isLoading, refresh }
}
