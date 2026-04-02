import { useState, useEffect, useCallback } from 'react'
import { resetBackendCredentials } from '../lib/backend'
import { ApiClient } from '../lib/api-client'
import { logger } from '../lib/logger'

interface BackendStatus {
  connected: boolean
  modelsLoaded: boolean
  gpuInfo: {
    name: string
    vram: number
    vramUsed: number
  } | null
}

export type BackendProcessStatus = 'alive' | 'restarting' | 'dead'

interface BackendHealthStatusPayload {
  status: BackendProcessStatus
  exitCode?: number | null
}

interface UseBackendReturn {
  status: BackendStatus
  processStatus: BackendProcessStatus | null
  isLoading: boolean
  error: string | null
  checkHealth: () => Promise<boolean>
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
  const [status, setStatus] = useState<BackendStatus>({
    connected: false,
    modelsLoaded: false,
    gpuInfo: null,
  })
  const [processStatus, setProcessStatus] = useState<BackendProcessStatus | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      logger.info('Checking backend health...')
      const data = await ApiClient.getHealth()
      logger.info(`Backend health: ${JSON.stringify(data)}`)

      setStatus({
        connected: true,
        modelsLoaded: data.models_loaded,
        gpuInfo: data.gpu_info,
      })
      setError(null)
      return true
    } catch (err) {
      logger.error(`Backend health check error: ${err}`)
      setStatus(prev => ({ ...prev, connected: false }))
      return false
    }
  }, [])

  const handleBackendStatus = useCallback(async (payload: BackendHealthStatusPayload) => {
    setProcessStatus(payload.status)

    if (payload.status === 'alive') {
      // Reset cached credentials so the new port/token are fetched
      resetBackendCredentials()
      const healthy = await checkHealth()
      if (!healthy) {
        setError('Failed to connect to backend')
      }
      setIsLoading(false)
      return
    }

    if (payload.status === 'restarting') {
      return
    }

    setStatus((prev) => ({ ...prev, connected: false }))
    setError('The backend process crashed and could not be restarted')
    setIsLoading(false)
  }, [checkHealth])

  useEffect(() => {
    let cancelled = false

    const applyStatus = async (value: unknown) => {
      const payload = toBackendHealthStatus(value)
      if (!payload || cancelled) {
        return
      }
      await handleBackendStatus(payload)
    }

    const unsubscribe = window.electronAPI.onBackendHealthStatus((data: BackendHealthStatusPayload) => {
      void applyStatus(data)
    })

    const init = async () => {
      try {
        const snapshot = await window.electronAPI.getBackendHealthStatus()
        await applyStatus(snapshot)
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
    status,
    processStatus,
    isLoading,
    error,
    checkHealth,
  }
}
