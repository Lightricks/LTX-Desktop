import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { resetBackendCredentials } from '../lib/backend'
import { ApiClient, type ApiSuccessOf } from '../lib/api-client'
import { useRemoteStatus } from '../hooks/use-remote-status'
import { isRemoteExposureEnabled, type RemoteExposure } from '../lib/remote-exposure'
import { shouldNotifyRemoteKeepAwake } from '../lib/remote-keep-awake-signal'

// promptEnhancerEnabled gates LTX API enhance_prompt (remote video and API text encoding) for
// every conditioning, including Gen Space. exploreAutoEnhancePrompts gates automatic rewrite on
// Home/Remote local Generate; Gen Space still rewrites locally and ignores that setting.
export interface AppSettings {
  useTorchCompile: boolean
  diffusionStageCacheEnabled: boolean
  hasLtxApiKey: boolean
  userPrefersLtxApiVideoGenerations: boolean
  hasFalApiKey: boolean
  userPrefersFalApiImageGenerations: boolean
  hasGeminiApiKey: boolean
  geminiModel: string
  useLocalTextEncoder: boolean
  promptCacheSize: number
  promptEnhancerEnabled: boolean
  // The user's explicit prompt-enhancer provider choice, persisted so it survives restarts.
  // null means no active choice yet — the enhancer defaults to whichever provider is available
  // without writing that default back here; only an explicit pick (never an automatic fallback
  // when the preferred provider is temporarily unavailable) sets this.
  promptEnhancerProviderPreference: 'local' | 'api' | null
  /** Home / Remote Explore: rewrite prompts automatically before Generate. */
  exploreAutoEnhancePrompts: boolean
  // The generation seed and its lock are deliberately not here: this state is re-POSTed to
  // /api/settings whole, so a stale copy would overwrite a seed another client just set.
  // They live behind useGenerationSeed (GET/POST /api/generation-seed).
  modelsDir: string
  useConvVae: boolean
  remoteExposure: RemoteExposure
}

export const DEFAULT_GEMINI_MODEL = 'gemini-3.5-flash-lite'

export const DEFAULT_APP_SETTINGS: AppSettings = {
  useTorchCompile: false,
  diffusionStageCacheEnabled: false,
  hasLtxApiKey: false,
  userPrefersLtxApiVideoGenerations: false,
  hasFalApiKey: false,
  userPrefersFalApiImageGenerations: false,
  hasGeminiApiKey: false,
  geminiModel: '',
  useLocalTextEncoder: false,
  promptCacheSize: 1,
  promptEnhancerEnabled: true,
  promptEnhancerProviderPreference: null,
  exploreAutoEnhancePrompts: true,
  modelsDir: '',
  useConvVae: false,
  remoteExposure: 'off',
}

type BackendProcessStatus = 'alive' | 'restarting' | 'dead'

interface AppSettingsContextValue {
  settings: AppSettings
  isLoaded: boolean
  runtimePolicyLoaded: boolean
  updateSettings: (patch: Partial<AppSettings> | ((prev: AppSettings) => AppSettings)) => void
  refreshSettings: () => Promise<void>
  saveLtxApiKey: (value: string) => Promise<void>
  saveFalApiKey: (value: string) => Promise<void>
  saveGeminiApiKey: (value: string) => Promise<void>
  clearLtxApiKey: () => Promise<void>
  clearFalApiKey: () => Promise<void>
  clearGeminiApiKey: () => Promise<void>
  forceApiGenerations: boolean
  localViable: boolean
  shouldVideoGenerateWithLtxApi: boolean
  shouldImageGenerateWithFalApi: boolean
  cudaAvailable: boolean
  // Bumped whenever installed models change (download / delete / activate a version). Generation
  // model specs are derived from the *active* local model, so anything reading them must refetch;
  // without this they stay pinned to whatever was installed at app start.
  modelsVersion: number
  notifyModelsChanged: () => void
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
    diffusionStageCacheEnabled: data.diffusionStageCacheEnabled ?? DEFAULT_APP_SETTINGS.diffusionStageCacheEnabled,
    hasLtxApiKey: data.hasLtxApiKey ?? DEFAULT_APP_SETTINGS.hasLtxApiKey,
    userPrefersLtxApiVideoGenerations: data.userPrefersLtxApiVideoGenerations ?? DEFAULT_APP_SETTINGS.userPrefersLtxApiVideoGenerations,
    hasFalApiKey: data.hasFalApiKey ?? DEFAULT_APP_SETTINGS.hasFalApiKey,
    userPrefersFalApiImageGenerations: data.userPrefersFalApiImageGenerations ?? DEFAULT_APP_SETTINGS.userPrefersFalApiImageGenerations,
    hasGeminiApiKey: data.hasGeminiApiKey ?? DEFAULT_APP_SETTINGS.hasGeminiApiKey,
    geminiModel: data.geminiModel ?? DEFAULT_APP_SETTINGS.geminiModel,
    useLocalTextEncoder: data.useLocalTextEncoder ?? DEFAULT_APP_SETTINGS.useLocalTextEncoder,
    promptCacheSize: data.promptCacheSize ?? DEFAULT_APP_SETTINGS.promptCacheSize,
    promptEnhancerEnabled: data.promptEnhancerEnabled ?? DEFAULT_APP_SETTINGS.promptEnhancerEnabled,
    promptEnhancerProviderPreference: data.promptEnhancerProviderPreference ?? DEFAULT_APP_SETTINGS.promptEnhancerProviderPreference,
    exploreAutoEnhancePrompts:
      data.exploreAutoEnhancePrompts ?? DEFAULT_APP_SETTINGS.exploreAutoEnhancePrompts,
    modelsDir: data.modelsDir ?? DEFAULT_APP_SETTINGS.modelsDir,
    useConvVae: data.useConvVae ?? DEFAULT_APP_SETTINGS.useConvVae,
    remoteExposure: data.remoteExposure === 'lan' ? 'lan' : DEFAULT_APP_SETTINGS.remoteExposure,
  }
}

type RuntimePolicyPayload = ApiSuccessOf<'getRuntimePolicy'>
type GpuInfoPayload = ApiSuccessOf<'getGpuInfo'>

export function AppSettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_APP_SETTINGS)
  const [isLoaded, setIsLoaded] = useState(false)
  const [runtimePolicyLoaded, setRuntimePolicyLoaded] = useState(false)
  const [forceApiGenerations, setForceApiGenerations] = useState(true)
  const [localViable, setLocalViable] = useState(false)
  const [cudaAvailable, setCudaAvailable] = useState(false)
  const [backendProcessStatus, setBackendProcessStatus] = useState<BackendProcessStatus | null>(null)
  const [modelsVersion, setModelsVersion] = useState(0)

  const notifyModelsChanged = useCallback(() => {
    setModelsVersion((current) => current + 1)
  }, [])

  useEffect(() => {
    if (backendProcessStatus !== 'alive') return

    let cancelled = false
    setRuntimePolicyLoaded(false)

    const fetchRuntimePolicy = async () => {
      const result = await ApiClient.getRuntimePolicy()
      if (!result.ok) {
        if (!cancelled) {
          // Fail closed until policy can be read. Do not send Explore to an LTX key.
          setForceApiGenerations(true)
          setLocalViable(false)
          setRuntimePolicyLoaded(true)
        }
        return
      }

      const payload = result.data as RuntimePolicyPayload
      if (typeof payload.force_api_generations !== 'boolean') {
        if (!cancelled) {
          setForceApiGenerations(true)
        }
      } else if (!cancelled) {
        setForceApiGenerations(payload.force_api_generations)
      }

      if (!cancelled) {
        setLocalViable(
          typeof payload.local_viable === 'boolean'
            ? payload.local_viable
            : payload.force_api_generations === false,
        )
        setRuntimePolicyLoaded(true)
      }
    }

    void fetchRuntimePolicy()

    return () => {
      cancelled = true
    }
  }, [backendProcessStatus])

  useEffect(() => {
    if (backendProcessStatus !== 'alive') return

    let cancelled = false

    const fetchGpuInfo = async () => {
      const result = await ApiClient.getGpuInfo()
      if (!result.ok || cancelled) return

      const payload = result.data as GpuInfoPayload
      setCudaAvailable(Boolean(payload.cuda_available))
    }

    void fetchGpuInfo()

    return () => {
      cancelled = true
    }
  }, [backendProcessStatus, modelsVersion])

  useEffect(() => {
    let cancelled = false

    const applyStatus = (value: unknown) => {
      const nextStatus = toBackendProcessStatus(value)
      if (!nextStatus || cancelled) {
        return
      }
      if (nextStatus === 'alive') {
        resetBackendCredentials()
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
    const result = await ApiClient.getSettings()
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    setSettings(normalizeAppSettings(result.data))
    setIsLoaded(true)
  }, [])

  useEffect(() => {
    if (isLoaded || backendProcessStatus !== 'alive') return

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
  }, [backendProcessStatus, isLoaded, refreshSettings])

  const remoteEnabled = isRemoteExposureEnabled(settings.remoteExposure)
  const backendAlive = backendProcessStatus === 'alive'
  const { status: remoteStatus } = useRemoteStatus(isLoaded && backendAlive && remoteEnabled)

  useEffect(() => {
    if (!isLoaded) return
    void window.electronAPI.notifyRemoteExposure({
      active: shouldNotifyRemoteKeepAwake({
        remoteEnabled,
        backendAlive,
        status: remoteStatus,
      }),
    })
  }, [backendAlive, isLoaded, remoteEnabled, remoteStatus])

  useEffect(() => {
    if (!isLoaded || backendProcessStatus !== 'alive') return
    const syncTimer = setTimeout(async () => {
      // Key fields are omitted: this sync must not write credentials. A key save
      // goes through writeApiKey, which already persisted the trimmed value.
      const { hasLtxApiKey: _a, hasFalApiKey: _b, hasGeminiApiKey: _c, modelsDir: _d, ...syncPayload } = settings
      const result = await ApiClient.updateSettings(syncPayload)
      if (!result.ok) {
        // Best-effort settings sync.
      }
    }, 150)
    return () => clearTimeout(syncTimer)
  }, [backendProcessStatus, isLoaded, settings])

  const updateSettings = useCallback((patch: Partial<AppSettings> | ((prev: AppSettings) => AppSettings)) => {
    if (typeof patch === 'function') {
      setSettings((prev) => patch(prev))
      return
    }
    setSettings((prev) => ({ ...prev, ...patch }))
  }, [])

  type ApiKeyField = 'ltxApiKey' | 'falApiKey' | 'geminiApiKey'

  const writeApiKey = useCallback(async (field: ApiKeyField, value: string) => {
    const patch =
      field === 'ltxApiKey' ? { ltxApiKey: value }
      : field === 'falApiKey' ? { falApiKey: value }
      : { geminiApiKey: value }
    const result = await ApiClient.updateSettings(patch)
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    if (value === '') {
      const flag =
        field === 'ltxApiKey' ? 'hasLtxApiKey'
        : field === 'falApiKey' ? 'hasFalApiKey'
        : 'hasGeminiApiKey'
      setSettings((prev) => ({ ...prev, [flag]: false }))
    }
    await refreshSettings()
  }, [refreshSettings])

  const saveLtxApiKey = useCallback((value: string) => writeApiKey('ltxApiKey', value), [writeApiKey])
  const saveFalApiKey = useCallback((value: string) => writeApiKey('falApiKey', value), [writeApiKey])
  const saveGeminiApiKey = useCallback((value: string) => writeApiKey('geminiApiKey', value), [writeApiKey])
  const clearLtxApiKey = useCallback(() => writeApiKey('ltxApiKey', ''), [writeApiKey])
  const clearFalApiKey = useCallback(() => writeApiKey('falApiKey', ''), [writeApiKey])
  const clearGeminiApiKey = useCallback(() => writeApiKey('geminiApiKey', ''), [writeApiKey])

  const shouldVideoGenerateWithLtxApi =
    settings.hasLtxApiKey &&
    (forceApiGenerations || settings.userPrefersLtxApiVideoGenerations)
  const shouldImageGenerateWithFalApi =
    settings.hasFalApiKey &&
    (forceApiGenerations || settings.userPrefersFalApiImageGenerations)

  const contextValue = useMemo<AppSettingsContextValue>(
    () => ({
      settings,
      isLoaded,
      runtimePolicyLoaded,
      updateSettings,
      refreshSettings,
      saveLtxApiKey,
      saveFalApiKey,
      saveGeminiApiKey,
      clearLtxApiKey,
      clearFalApiKey,
      clearGeminiApiKey,
      forceApiGenerations,
      localViable,
      shouldVideoGenerateWithLtxApi,
      shouldImageGenerateWithFalApi,
      cudaAvailable,
      modelsVersion,
      notifyModelsChanged,
    }),
    [cudaAvailable, forceApiGenerations, isLoaded, localViable, modelsVersion, notifyModelsChanged, refreshSettings, runtimePolicyLoaded, clearFalApiKey, clearGeminiApiKey, clearLtxApiKey, saveFalApiKey, saveGeminiApiKey, saveLtxApiKey, settings, shouldVideoGenerateWithLtxApi, shouldImageGenerateWithFalApi, updateSettings],
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
