import { useState, useEffect, useCallback } from 'react'
import { resetBackendCredentials } from '../lib/backend'
import { logger } from '../lib/logger'

export type BackendProcessStatus = 'connecting' | 'alive' | 'restarting' | 'unreachable' | 'dead'
export type BackendConnectionMode = 'managed-local' | 'external'

interface BackendHealthStatusPayload {
  mode: BackendConnectionMode
  status: BackendProcessStatus
  exitCode?: number | null
  checkedAt: number
  message?: string
}

interface UseBackendReturn {
  processStatus: BackendProcessStatus | null
  connected: boolean
  isLoading: boolean
  connectionMode: BackendConnectionMode | null
  message: string | null
}

function toBackendHealthStatus(value: unknown): BackendHealthStatusPayload | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const record = value as { mode?: unknown; status?: unknown; exitCode?: unknown; checkedAt?: unknown; message?: unknown }
  if (record.mode !== 'managed-local' && record.mode !== 'external') {
    return null
  }
  if (
    record.status !== 'connecting'
    && record.status !== 'alive'
    && record.status !== 'restarting'
    && record.status !== 'unreachable'
    && record.status !== 'dead'
  ) {
    return null
  }

  return {
    mode: record.mode,
    status: record.status,
    exitCode: typeof record.exitCode === 'number' || record.exitCode === null ? record.exitCode : undefined,
    checkedAt: typeof record.checkedAt === 'number' ? record.checkedAt : Date.now(),
    message: typeof record.message === 'string' ? record.message : undefined,
  }
}

export function useBackend(): UseBackendReturn {
  const [processStatus, setProcessStatus] = useState<BackendProcessStatus | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [connectionMode, setConnectionMode] = useState<BackendConnectionMode | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const handleBackendStatus = useCallback((payload: BackendHealthStatusPayload) => {
    setProcessStatus(payload.status)
    setConnectionMode(payload.mode)
    setMessage(payload.message ?? null)

    if (payload.status === 'alive') {
      // Main has verified HTTP reachability before publishing 'alive' and may
      // have spawned a fresh backend with a new port/token — drop cached creds
      // so the next backendFetch picks up the current values.
      resetBackendCredentials()
      setIsLoading(false)
      return
    }

    if (payload.status === 'restarting' || payload.status === 'connecting') {
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
    connectionMode,
    message,
  }
}
