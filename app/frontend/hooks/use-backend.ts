import { useState, useEffect, useCallback } from 'react'
import { resetBackendCredentials } from '../lib/backend'
import { logger } from '../lib/logger'

export type BackendProcessStatus = 'alive' | 'restarting' | 'dead'

interface BackendHealthStatusPayload {
  status: BackendProcessStatus
  exitCode?: number | null
}

interface UseBackendReturn {
  processStatus: BackendProcessStatus | null
  connected: boolean
  isLoading: boolean
}

function toBackendHealthStatus(value: unknown): BackendHealthStatusPayload | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const record = value as { status?: unknown; exitCode?: unknown }
  if (record.status !== 'alive' && record.status !== 'restarting' && record.status !== 'dead') {
    return null
  }

  return {
    status: record.status,
    exitCode: typeof record.exitCode === 'number' || record.exitCode === null ? record.exitCode : undefined,
  }
}

export function useBackend(): UseBackendReturn {
  const [processStatus, setProcessStatus] = useState<BackendProcessStatus | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const handleBackendStatus = useCallback((payload: BackendHealthStatusPayload) => {
    setProcessStatus(payload.status)

    if (payload.status === 'alive') {
      // Main has verified HTTP reachability before publishing 'alive' and may
      // have spawned a fresh backend with a new port/token — drop cached creds
      // so the next backendFetch picks up the current values.
      resetBackendCredentials()
      setIsLoading(false)
      return
    }

    if (payload.status === 'restarting') {
      return
    }

    setIsLoading(false)
  }, [])

  useEffect(() => {
    let cancelled = false

    const applyStatus = (value: unknown) => {
      const payload = toBackendHealthStatus(value)
      if (!payload || cancelled) {
        return
      }
      handleBackendStatus(payload)
    }

    const unsubscribe = window.electronAPI.onBackendHealthStatus((data: BackendHealthStatusPayload) => {
      applyStatus(data)
    })

    const init = async () => {
      try {
        const snapshot = await window.electronAPI.getBackendHealthStatus()
        applyStatus(snapshot)
      } catch (err) {
        logger.error(`Failed to load backend health status snapshot: ${err}`)
      }
    }

    void init()

    return () => {
      cancelled = true
      unsubscribe()
    }
  }, [handleBackendStatus])

  return {
    processStatus,
    connected: processStatus === 'alive',
    isLoading,
  }
}
