import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'

export interface CreditPricing {
  video_t2v: number
  video_i2v: number
  video_seedance: number
  image: number
  image_edit: number
  audio: number
  text_enhance: number
}

export interface CreditInfo {
  balance_cents: number | null
  pricing: CreditPricing | null
}

export interface InferenceSettings {
  steps: number
  useUpscaler: boolean
}

export interface FastModelSettings {
  useUpscaler: boolean
}

export interface AppSettings {
  useTorchCompile: boolean
  loadOnStartup: boolean
  hasLtxApiKey: boolean
  userPrefersLtxApiVideoGenerations: boolean
  hasReplicateApiKey: boolean
  hasPaletteApiKey: boolean
  imageModel: string
  videoModel: string
  hasGeminiApiKey: boolean
  useLocalTextEncoder: boolean
  fastModel: FastModelSettings
  proModel: InferenceSettings
  promptCacheSize: number
  promptEnhancerEnabledT2V: boolean
  promptEnhancerEnabledI2V: boolean
  seedLocked: boolean
  lockedSeed: number
  hasCivitaiApiKey: boolean
  customVideoModelPath: string
  selectedVideoModel: string
}

export const DEFAULT_APP_SETTINGS: AppSettings = {
  useTorchCompile: false,
  loadOnStartup: true,
  hasLtxApiKey: false,
  userPrefersLtxApiVideoGenerations: false,
  hasReplicateApiKey: false,
  hasPaletteApiKey: false,
  imageModel: 'flux-klein-9b',
  videoModel: 'ltx-fast',
  hasGeminiApiKey: false,
  useLocalTextEncoder: false,
  fastModel: { useUpscaler: true },
  proModel: { steps: 20, useUpscaler: true },
  promptCacheSize: 1,
  promptEnhancerEnabledT2V: false,
  promptEnhancerEnabledI2V: false,
  seedLocked: false,
  lockedSeed: 42,
  hasCivitaiApiKey: false,
  customVideoModelPath: '',
  selectedVideoModel: '',
}

type BackendProcessStatus = 'alive' | 'restarting' | 'dead'

interface AppSettingsContextValue {
  settings: AppSettings
  isLoaded: boolean
  runtimePolicyLoaded: boolean
  updateSettings: (patch: Partial<AppSettings> | ((prev: AppSettings) => AppSettings)) => void
  refreshSettings: () => Promise<void>
  saveLtxApiKey: (value: string) => Promise<void>
  saveReplicateApiKey: (value: string) => Promise<void>
  saveGeminiApiKey: (value: string) => Promise<void>
  savePaletteApiKey: (value: string) => Promise<void>
  saveCivitaiApiKey: (value: string) => Promise<void>
  forceApiGenerations: boolean
  shouldVideoGenerateWithLtxApi: boolean
  credits: CreditInfo
  refreshCredits: () => Promise<void>
}

const AppSettingsContext = createContext<AppSettingsContextValue | null>(null)

function toBackendProcessStatus(value: unknown): BackendProcessStatus | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const record = value as { status?: unknown }
  if (record.status === 'alive' || record.status === 'restarting' || record.status === 'dead') {
    return record.status
  }
  return null
}

function normalizeAppSettings(data: Partial<AppSettings>): AppSettings {
  return {
    useTorchCompile: data.useTorchCompile ?? DEFAULT_APP_SETTINGS.useTorchCompile,
    loadOnStartup: data.loadOnStartup ?? DEFAULT_APP_SETTINGS.loadOnStartup,
    hasLtxApiKey: data.hasLtxApiKey ?? DEFAULT_APP_SETTINGS.hasLtxApiKey,
    userPrefersLtxApiVideoGenerations: data.userPrefersLtxApiVideoGenerations ?? DEFAULT_APP_SETTINGS.userPrefersLtxApiVideoGenerations,
    hasReplicateApiKey: data.hasReplicateApiKey ?? DEFAULT_APP_SETTINGS.hasReplicateApiKey,
    hasPaletteApiKey: data.hasPaletteApiKey ?? DEFAULT_APP_SETTINGS.hasPaletteApiKey,
    imageModel: data.imageModel ?? DEFAULT_APP_SETTINGS.imageModel,
    videoModel: data.videoModel ?? DEFAULT_APP_SETTINGS.videoModel,
    hasGeminiApiKey: data.hasGeminiApiKey ?? DEFAULT_APP_SETTINGS.hasGeminiApiKey,
    useLocalTextEncoder: data.useLocalTextEncoder ?? DEFAULT_APP_SETTINGS.useLocalTextEncoder,
    fastModel: data.fastModel ?? DEFAULT_APP_SETTINGS.fastModel,
    proModel: data.proModel ?? DEFAULT_APP_SETTINGS.proModel,
    promptCacheSize: data.promptCacheSize ?? DEFAULT_APP_SETTINGS.promptCacheSize,
    promptEnhancerEnabledT2V: data.promptEnhancerEnabledT2V ?? DEFAULT_APP_SETTINGS.promptEnhancerEnabledT2V,
    promptEnhancerEnabledI2V: data.promptEnhancerEnabledI2V ?? DEFAULT_APP_SETTINGS.promptEnhancerEnabledI2V,
    seedLocked: data.seedLocked ?? DEFAULT_APP_SETTINGS.seedLocked,
    lockedSeed: data.lockedSeed ?? DEFAULT_APP_SETTINGS.lockedSeed,
    hasCivitaiApiKey: data.hasCivitaiApiKey ?? DEFAULT_APP_SETTINGS.hasCivitaiApiKey,
    customVideoModelPath: data.customVideoModelPath ?? DEFAULT_APP_SETTINGS.customVideoModelPath,
    selectedVideoModel: data.selectedVideoModel ?? DEFAULT_APP_SETTINGS.selectedVideoModel,
  }
}

export function AppSettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_APP_SETTINGS)
  const [isLoaded, setIsLoaded] = useState(false)
  const [runtimePolicyLoaded, setRuntimePolicyLoaded] = useState(false)
  const [backendUrl, setBackendUrl] = useState<string | null>(null)
  const [forceApiGenerations, setForceApiGenerations] = useState(true)
  const [backendProcessStatus, setBackendProcessStatus] = useState<BackendProcessStatus | null>(null)
  const [credits, setCredits] = useState<CreditInfo>({ balance_cents: null, pricing: null })
  const creditsPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    window.electronAPI.getBackendUrl().then(setBackendUrl).catch(() => setBackendUrl(null))
  }, [])

  useEffect(() => {
    if (!backendUrl || backendProcessStatus !== 'alive') return

    let cancelled = false
    setRuntimePolicyLoaded(false)

    const fetchRuntimePolicy = async () => {
      try {
        const response = await fetch(`${backendUrl}/api/runtime-policy`)
        if (!response.ok) {
          throw new Error(`Runtime policy fetch failed with status ${response.status}`)
        }

        const payload = (await response.json()) as { force_api_generations?: unknown }
        if (typeof payload.force_api_generations !== 'boolean') {
          throw new Error('Runtime policy response missing force_api_generations boolean')
        }

        if (!cancelled) {
          setForceApiGenerations(payload.force_api_generations)
        }
      } catch {
        if (!cancelled) {
          // Fail closed until policy can be read.
          setForceApiGenerations(true)
        }
      } finally {
        if (!cancelled) {
          setRuntimePolicyLoaded(true)
        }
      }
    }

    void fetchRuntimePolicy()

    return () => {
      cancelled = true
    }
  }, [backendProcessStatus, backendUrl])

  useEffect(() => {
    let cancelled = false

    const applyStatus = (value: unknown) => {
      const nextStatus = toBackendProcessStatus(value)
      if (!nextStatus || cancelled) {
        return
      }
      setBackendProcessStatus(nextStatus)
    }

    const unsubscribe = window.electronAPI.onBackendHealthStatus((data) => {
      applyStatus(data)
    })

    void window.electronAPI.getBackendHealthStatus()
      .then((snapshot) => {
        applyStatus(snapshot)
      })
      .catch(() => {
        // Snapshot is optional at startup; subscription continues to listen for pushes.
      })

    return () => {
      cancelled = true
      unsubscribe()
    }
  }, [])

  const refreshSettings = useCallback(async () => {
    if (!backendUrl) return
    const response = await fetch(`${backendUrl}/api/settings`)
    if (!response.ok) {
      throw new Error(`Settings fetch failed with status ${response.status}`)
    }
    const data = await response.json()
    setSettings(normalizeAppSettings(data))
    setIsLoaded(true)
  }, [backendUrl])

  useEffect(() => {
    if (!backendUrl || isLoaded || backendProcessStatus !== 'alive') return

    let cancelled = false
    let retryTimer: ReturnType<typeof setTimeout> | null = null

    const fetchSettings = async () => {
      try {
        await refreshSettings()
        if (cancelled) return
      } catch {
        if (!cancelled) {
          retryTimer = setTimeout(fetchSettings, 1000)
        }
      }
    }

    fetchSettings()

    return () => {
      cancelled = true
      if (retryTimer) clearTimeout(retryTimer)
    }
  }, [backendProcessStatus, backendUrl, isLoaded, refreshSettings])

  useEffect(() => {
    if (!backendUrl || !isLoaded || backendProcessStatus !== 'alive') return
    const syncTimer = setTimeout(async () => {
      try {
        const { hasLtxApiKey: _a, hasReplicateApiKey: _b, hasGeminiApiKey: _c, hasPaletteApiKey: _d, ...syncPayload } = settings
        await fetch(`${backendUrl}/api/settings`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(syncPayload),
        })
      } catch {
        // Best-effort settings sync.
      }
    }, 150)
    return () => clearTimeout(syncTimer)
  }, [backendProcessStatus, backendUrl, isLoaded, settings])

  const updateSettings = useCallback((patch: Partial<AppSettings> | ((prev: AppSettings) => AppSettings)) => {
    if (typeof patch === 'function') {
      setSettings((prev) => patch(prev))
      return
    }
    setSettings((prev) => ({ ...prev, ...patch }))
  }, [])

  const saveLtxApiKey = useCallback(async (value: string) => {
    if (!backendUrl) return
    const response = await fetch(`${backendUrl}/api/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ltxApiKey: value }),
    })
    if (!response.ok) {
      const detail = await response.text()
      throw new Error(detail || 'Failed to save LTX API key.')
    }
    await refreshSettings()
  }, [backendUrl, refreshSettings])

  const saveGeminiApiKey = useCallback(async (value: string) => {
    if (!backendUrl) return
    const response = await fetch(`${backendUrl}/api/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ geminiApiKey: value }),
    })
    if (!response.ok) {
      const detail = await response.text()
      throw new Error(detail || 'Failed to save Gemini API key.')
    }
    await refreshSettings()
  }, [backendUrl, refreshSettings])

  const saveCivitaiApiKey = useCallback(async (value: string) => {
    if (!backendUrl) return
    const response = await fetch(`${backendUrl}/api/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ civitaiApiKey: value }),
    })
    if (!response.ok) {
      const detail = await response.text()
      throw new Error(detail || 'Failed to save CivitAI API key.')
    }
    await refreshSettings()
  }, [backendUrl, refreshSettings])

  const saveReplicateApiKey = useCallback(async (value: string) => {
    if (!backendUrl) return
    const response = await fetch(`${backendUrl}/api/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ replicateApiKey: value }),
    })
    if (!response.ok) {
      const detail = await response.text()
      throw new Error(detail || 'Failed to save Replicate API key.')
    }
    await refreshSettings()
  }, [backendUrl, refreshSettings])

  const savePaletteApiKey = useCallback(async (value: string) => {
    if (!backendUrl) return
    const response = await fetch(`${backendUrl}/api/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paletteApiKey: value }),
    })
    if (!response.ok) {
      const detail = await response.text()
      throw new Error(detail || 'Failed to save Palette API key.')
    }
    await refreshSettings()
  }, [backendUrl, refreshSettings])

  const refreshCredits = useCallback(async () => {
    if (!backendUrl) return
    try {
      const res = await fetch(`${backendUrl}/api/sync/credits`)
      if (!res.ok) return
      const data = await res.json()
      if (data.connected) {
        setCredits({
          balance_cents: data.balance_cents ?? null,
          pricing: data.pricing ?? null,
        })
      } else {
        setCredits({ balance_cents: null, pricing: null })
      }
    } catch { /* ignore */ }
  }, [backendUrl])

  // Poll credits every 30s when connected
  useEffect(() => {
    if (!backendUrl || backendProcessStatus !== 'alive' || !settings.hasPaletteApiKey) {
      setCredits({ balance_cents: null, pricing: null })
      if (creditsPollRef.current) {
        clearInterval(creditsPollRef.current)
        creditsPollRef.current = null
      }
      return
    }
    // Fetch immediately, then poll
    void refreshCredits()
    creditsPollRef.current = setInterval(() => void refreshCredits(), 30_000)
    return () => {
      if (creditsPollRef.current) {
        clearInterval(creditsPollRef.current)
        creditsPollRef.current = null
      }
    }
  }, [backendUrl, backendProcessStatus, settings.hasPaletteApiKey, refreshCredits])

  const shouldVideoGenerateWithLtxApi =
    forceApiGenerations || (settings.userPrefersLtxApiVideoGenerations && settings.hasLtxApiKey)

  const contextValue = useMemo<AppSettingsContextValue>(
    () => ({
      settings,
      isLoaded,
      runtimePolicyLoaded,
      updateSettings,
      refreshSettings,
      saveLtxApiKey,
      saveReplicateApiKey,
      saveGeminiApiKey,
      savePaletteApiKey,
      saveCivitaiApiKey,
      forceApiGenerations,
      shouldVideoGenerateWithLtxApi,
      credits,
      refreshCredits,
    }),
    [credits, forceApiGenerations, isLoaded, refreshCredits, refreshSettings, runtimePolicyLoaded, saveCivitaiApiKey, savePaletteApiKey, saveReplicateApiKey, saveGeminiApiKey, saveLtxApiKey, settings, shouldVideoGenerateWithLtxApi, updateSettings],
  )

  return <AppSettingsContext.Provider value={contextValue}>{children}</AppSettingsContext.Provider>
}

export function useAppSettings() {
  const context = useContext(AppSettingsContext)
  if (!context) {
    throw new Error('useAppSettings must be used within AppSettingsProvider')
  }
  return context
}
