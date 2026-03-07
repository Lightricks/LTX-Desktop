import { useState, useEffect, useCallback } from 'react'
import { backendFetch, backendWsUrl, resetBackendCredentials } from '../lib/backend'
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

interface ModelStatus {
  id: string
  name: string
  size: number
  downloaded: boolean
  downloadProgress: number
}

export type BackendProcessStatus = 'alive' | 'restarting' | 'dead'

interface BackendHealthStatusPayload {
  status: BackendProcessStatus
  exitCode?: number | null
}

interface UseBackendReturn {
  status: BackendStatus
  models: ModelStatus[]
  processStatus: BackendProcessStatus | null
  isLoading: boolean
  error: string | null
  checkHealth: () => Promise<boolean>
  downloadModel: (modelId: string) => Promise<void>
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
  const [models, setModels] = useState<ModelStatus[]>([])
  const [processStatus, setProcessStatus] = useState<BackendProcessStatus | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      logger.info('Checking backend health...')
      const response = await backendFetch('/health')

      if (response.ok) {
        const data = await response.json()
        logger.info(`Backend health: ${JSON.stringify(data)}`)

        setStatus({
          connected: true,
          modelsLoaded: data.models_loaded,
          gpuInfo: data.gpu_info,
        })
        setError(null)
        return true
      }
      logger.warn(`Backend health check failed with status: ${response.status}`)
      return false
    } catch (err) {
      logger.error(`Backend health check error: ${err}`)
      setStatus(prev => ({ ...prev, connected: false }))
      return false
    }
  }, [])

  const fetchModels = useCallback(async () => {
    try {
      const response = await backendFetch('/api/models')

      if (response.ok) {
        const data = await response.json()
        setModels(data.models)
      }
    } catch (err) {
      logger.error(`Failed to fetch models: ${err}`)
    }
  }, [])

  const downloadModel = useCallback(async (modelId: string) => {
    try {
      // Connect to WebSocket for download progress
      const wsUrl = await backendWsUrl(`/ws/download/${modelId}`)
      const ws = new WebSocket(wsUrl)

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        if (data.type === 'progress') {
          setModels(prev => prev.map(m =>
            m.id === modelId
              ? { ...m, downloadProgress: data.progress }
              : m
          ))
        } else if (data.type === 'complete') {
          setModels(prev => prev.map(m =>
            m.id === modelId
              ? { ...m, downloaded: true, downloadProgress: 100 }
              : m
          ))
        }
      }

      // Trigger download
      await backendFetch(`/api/models/${modelId}/download`, {
        method: 'POST',
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed')
    }
  }, [])

  const handleBackendStatus = useCallback(async (payload: BackendHealthStatusPayload) => {
    setProcessStatus(payload.status)

    if (payload.status === 'alive') {
      // Reset cached credentials so the new port/token are fetched
      resetBackendCredentials()
      const healthy = await checkHealth()
      if (healthy) {
        await fetchModels()
      } else {
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
  }, [checkHealth, fetchModels])

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
    models,
    processStatus,
    isLoading,
    error,
    checkHealth,
    downloadModel,
  }
}
