import { useCallback, useEffect, useRef, useState } from 'react'

import type { DownloadProgressSnapshot } from '@/lib/download-progress'
import { logger } from '@/lib/logger'

interface SetupProgress {
  status: 'downloading' | 'extracting' | 'complete' | 'error'
  percent: number
  downloadedBytes: number
  totalBytes: number
  speed: number
}

// The main process pushes untyped progress payloads; ignore malformed ones instead
// of rendering NaN stats or dropping a terminal state (stuck-spinner fix).
function isSetupProgress(data: unknown): data is SetupProgress {
  if (typeof data !== 'object' || data === null) return false
  const progress = data as Record<string, unknown>
  return (
    (progress.status === 'downloading' ||
      progress.status === 'extracting' ||
      progress.status === 'complete' ||
      progress.status === 'error') &&
    typeof progress.percent === 'number' &&
    typeof progress.downloadedBytes === 'number' &&
    typeof progress.totalBytes === 'number' &&
    typeof progress.speed === 'number'
  )
}

/** Starts the Windows Python runtime download while `active`. The splash owns the UI. */
export function usePythonSetup({ active }: { active: boolean }): {
  snapshot: DownloadProgressSnapshot | null
  error: string | null
  retry: () => void
} {
  const [progress, setProgress] = useState<SetupProgress | null>(null)
  const [error, setError] = useState<string | null>(null)
  const started = useRef(false)

  const startSetup = useCallback(async () => {
    setError(null)
    try {
      await window.electronAPI.startPythonSetup()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to download Python environment.')
    }
  }, [])

  useEffect(() => {
    if (!active) {
      started.current = false
      setProgress(null)
      setError(null)
      return
    }
    window.electronAPI.onPythonSetupProgress((data: unknown) => {
      if (!isSetupProgress(data)) {
        logger.error('Ignoring malformed python-setup-progress payload.')
        return
      }
      setProgress(data)
      if (data.status === 'error') {
        setError('Download failed. Please check your internet connection and try again.')
      }
    })
    return () => {
      window.electronAPI.removePythonSetupProgress()
    }
  }, [active])

  useEffect(() => {
    if (!active || started.current) return
    started.current = true
    void startSetup()
  }, [active, startSetup])

  const retry = useCallback(() => {
    setError(null)
    started.current = false
    void startSetup()
  }, [startSetup])

  if (!active) {
    return { snapshot: null, error, retry }
  }

  const extracting = progress?.status === 'extracting'
  return {
    snapshot: {
      statusLabel: extracting ? 'Extracting Python' : 'Downloading Python',
      percent: progress?.percent ?? 0,
      downloadedBytes: progress?.downloadedBytes ?? 0,
      totalBytes: progress?.totalBytes ?? 0,
      speedBytesPerSec: progress?.speed ?? 0,
    },
    error,
    retry,
  }
}
