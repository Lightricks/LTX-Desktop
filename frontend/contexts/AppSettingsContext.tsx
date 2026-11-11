import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { resetBackendCredentials } from '../lib/backend'
import { ApiClient, type ApiSuccessOf } from '../lib/api-client'

type AssistantSelection = ApiSuccessOf<'getAssistantSelection'>
import { useRemoteStatus } from '../hooks/use-remote-status'
import { isRemoteExposureEnabled, type RemoteExposure } from '../lib/remote-exposure'
import { shouldNotifyRemoteKeepAwake } from '../lib/remote-keep-awake-signal'
import { readBudgetMode, type BudgetMode } from '../lib/budget-risk'
import {
  CAUTIOUS_OFF_SAVE_ERROR,
  CAUTIOUS_ON_SAVE_ERROR,
  NEVER_SPLIT_OFF_SAVE_ERROR,
  NEVER_SPLIT_ON_SAVE_ERROR,
} from '../lib/settings-sync'

type BackendSettings = ApiSuccessOf<'getSettings'>

// promptEnhancerEnabled gates LTX API enhance_prompt (remote video and API text encoding) for
// every conditioning, including Gen Space. exploreAutoEnhancePrompts gates automatic rewrite on
// Home/Remote local Generate; Gen Space still rewrites locally and ignores that setting.
export interface AppSettings {
  useTorchCompile: boolean
  diffusionStageCacheEnabled: boolean
  mpsConv3DAsConv2DEnabled: BackendSettings['mpsConv3DAsConv2DEnabled']
  hasLtxApiKey: boolean
  userPrefersLtxApiVideoGenerations: boolean
  hasFalApiKey: boolean
  userPrefersFalApiImageGenerations: boolean
  hasGeminiApiKey: boolean
  geminiModel: string
  assistantProvider: BackendSettings['assistantProvider']
  assistantLocalModel: BackendSettings['assistantLocalModel']
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
  /** Shows the Advanced section of Refine Details. Other features ignore it for now. */
  advancedControls: boolean
  // The generation seed and its lock are deliberately not here: this state is re-POSTed to
  // /api/settings whole, so a stale copy would overwrite a seed another client just set.
  // They live behind useGenerationSeed (GET/POST /api/generation-seed).
  modelsDir: string
  useConvVae: boolean
  nagScale: number
  /** Show the per-run NAG strength row under the negative prompt (Desktop and paired devices). */
  showNagStrengthControls: boolean
  remoteExposure: RemoteExposure
  /** Brave runs over-budget and untested jobs after a confirm; Cautious refuses them. */
  budgetMode: BudgetMode
  neverSplitIntoParts: boolean
  /** Brave's "don't ask again": generate risky jobs without the confirm. */
  skipBudgetRiskConfirm: boolean
}

export { DEFAULT_GEMINI_MODEL } from '../lib/gemini-model'

export const DEFAULT_APP_SETTINGS: AppSettings = {
  useTorchCompile: false,
  diffusionStageCacheEnabled: false,
  mpsConv3DAsConv2DEnabled: false,
  hasLtxApiKey: false,
  userPrefersLtxApiVideoGenerations: false,
  hasFalApiKey: false,
  userPrefersFalApiImageGenerations: false,
  hasGeminiApiKey: false,
  geminiModel: '',
  assistantProvider: 'gemini',
  assistantLocalModel: 'qwen3.5:9b',
  useLocalTextEncoder: false,
  promptCacheSize: 1,
  promptEnhancerEnabled: true,
  promptEnhancerProviderPreference: null,
  exploreAutoEnhancePrompts: true,
  advancedControls: false,
  modelsDir: '',
  useConvVae: false,
  nagScale: 11,
  showNagStrengthControls: false,
  remoteExposure: 'off',
  budgetMode: 'brave',
  neverSplitIntoParts: false,
  skipBudgetRiskConfirm: false,
}

type BackendProcessStatus = 'alive' | 'restarting' | 'dead'

const ASSISTANT_SELECTION_REFRESH_MS = 10_000

interface AppSettingsContextValue {
  settings: AppSettings
  isLoaded: boolean
  runtimePolicyLoaded: boolean
  updateSettings: (patch: Partial<AppSettings> | ((prev: AppSettings) => AppSettings)) => void
  refreshSettings: () => Promise<void>
  /** Sets the assistant model Desktop Settings and both composers share. */
  selectAssistant: (provider: AppSettings['assistantProvider'], localModel?: AppSettings['assistantLocalModel']) => Promise<void>
  /** True while a selectAssistant save is in flight; a send now would use the previous model. */
  assistantSelectionPending: boolean
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
  // Saves the budget mode on its own and updates `settings.budgetMode` only after
  // the server accepts it. The document sync does not write this field.
  setBudgetMode: (mode: BudgetMode) => Promise<void>
  budgetModeSaving: boolean
  // Same as the budget mode: specs are priced with it, so it saves on its own.
  setNeverSplitIntoParts: (neverSplit: boolean) => Promise<void>
  neverSplitSaving: boolean
  settingsSyncError: string | null
  clearSettingsSyncError: () => void
  // Gemini model has its own save, like an API key. The document sync does not write it.
  saveGeminiModel: (model: string) => Promise<boolean>
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
    mpsConv3DAsConv2DEnabled: data.mpsConv3DAsConv2DEnabled ?? DEFAULT_APP_SETTINGS.mpsConv3DAsConv2DEnabled,
    hasLtxApiKey: data.hasLtxApiKey ?? DEFAULT_APP_SETTINGS.hasLtxApiKey,
    userPrefersLtxApiVideoGenerations: data.userPrefersLtxApiVideoGenerations ?? DEFAULT_APP_SETTINGS.userPrefersLtxApiVideoGenerations,
    hasFalApiKey: data.hasFalApiKey ?? DEFAULT_APP_SETTINGS.hasFalApiKey,
    userPrefersFalApiImageGenerations: data.userPrefersFalApiImageGenerations ?? DEFAULT_APP_SETTINGS.userPrefersFalApiImageGenerations,
    hasGeminiApiKey: data.hasGeminiApiKey ?? DEFAULT_APP_SETTINGS.hasGeminiApiKey,
    geminiModel: data.geminiModel ?? DEFAULT_APP_SETTINGS.geminiModel,
    assistantProvider: data.assistantProvider ?? DEFAULT_APP_SETTINGS.assistantProvider,
    assistantLocalModel: data.assistantLocalModel ?? DEFAULT_APP_SETTINGS.assistantLocalModel,
    useLocalTextEncoder: data.useLocalTextEncoder ?? DEFAULT_APP_SETTINGS.useLocalTextEncoder,
    promptCacheSize: data.promptCacheSize ?? DEFAULT_APP_SETTINGS.promptCacheSize,
    promptEnhancerEnabled: data.promptEnhancerEnabled ?? DEFAULT_APP_SETTINGS.promptEnhancerEnabled,
    promptEnhancerProviderPreference: data.promptEnhancerProviderPreference ?? DEFAULT_APP_SETTINGS.promptEnhancerProviderPreference,
    exploreAutoEnhancePrompts:
      data.exploreAutoEnhancePrompts ?? DEFAULT_APP_SETTINGS.exploreAutoEnhancePrompts,
    advancedControls: data.advancedControls ?? DEFAULT_APP_SETTINGS.advancedControls,
    modelsDir: data.modelsDir ?? DEFAULT_APP_SETTINGS.modelsDir,
    useConvVae: data.useConvVae ?? DEFAULT_APP_SETTINGS.useConvVae,
    nagScale: data.nagScale ?? DEFAULT_APP_SETTINGS.nagScale,
    showNagStrengthControls:
      data.showNagStrengthControls ?? DEFAULT_APP_SETTINGS.showNagStrengthControls,
    remoteExposure: data.remoteExposure === 'lan' ? 'lan' : DEFAULT_APP_SETTINGS.remoteExposure,
    budgetMode: readBudgetMode(data.budgetMode),
    neverSplitIntoParts: data.neverSplitIntoParts ?? DEFAULT_APP_SETTINGS.neverSplitIntoParts,
    skipBudgetRiskConfirm: data.skipBudgetRiskConfirm ?? DEFAULT_APP_SETTINGS.skipBudgetRiskConfirm,
  }
}

type RuntimePolicyPayload = ApiSuccessOf<'getRuntimePolicy'>
type GpuInfoPayload = ApiSuccessOf<'getGpuInfo'>

export function AppSettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_APP_SETTINGS)
  const settingsRef = useRef(settings)
  settingsRef.current = settings
  const budgetModeSavingRef = useRef(false)
  const budgetModeRevisionRef = useRef(0)
  const [budgetModeSaving, setBudgetModeSaving] = useState(false)
  const neverSplitSavingRef = useRef(false)
  const neverSplitRevisionRef = useRef(0)
  const [neverSplitSaving, setNeverSplitSaving] = useState(false)
  const [settingsSyncError, setSettingsSyncError] = useState<string | null>(null)
  const [isLoaded, setIsLoaded] = useState(false)
  const isLoadedRef = useRef(false)
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
    const revision = budgetModeRevisionRef.current
    const neverSplitRevision = neverSplitRevisionRef.current
    const loaded = isLoadedRef.current
    const result = await ApiClient.getSettings()
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    const server = normalizeAppSettings(result.data)
    isLoadedRef.current = true
    setIsLoaded(true)
    setSettings((prev) => {
      // After the first load, a reload only copies fields that have their own
      // writer. Document edits stay put, so this GET cannot put an older
      // checkbox or an older Gemini model back.
      const next = loaded
        ? {
            ...prev,
            hasLtxApiKey: server.hasLtxApiKey,
            hasFalApiKey: server.hasFalApiKey,
            hasGeminiApiKey: server.hasGeminiApiKey,
            modelsDir: server.modelsDir,
            geminiModel: server.geminiModel,
            budgetMode: server.budgetMode,
            neverSplitIntoParts: server.neverSplitIntoParts,
          }
        : server
      // A reload that overlaps the budget-mode request, or that started before
      // that request finished, must not move the switch.
      if (budgetModeSavingRef.current || budgetModeRevisionRef.current !== revision) {
        next.budgetMode = prev.budgetMode
      }
      if (neverSplitSavingRef.current || neverSplitRevisionRef.current !== neverSplitRevision) {
        next.neverSplitIntoParts = prev.neverSplitIntoParts
      }
      settingsRef.current = next
      return next
    })
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
    const syncTimer = setTimeout(() => {
      // Omitted fields have their own writer. A document save must not put an
      // older key flag, models dir, budget mode, Never split, or Gemini model back. The assistant
      // choice is omitted too: Remote can change it, and this would write a stale copy back.
      const {
        hasLtxApiKey: _a,
        hasFalApiKey: _b,
        hasGeminiApiKey: _c,
        modelsDir: _d,
        budgetMode: _e,
        neverSplitIntoParts: _i,
        geminiModel: _f,
        assistantProvider: _g,
        assistantLocalModel: _h,
        ...syncPayload
      } = settings
      void ApiClient.updateSettings(syncPayload).then((result) => {
        if (result.ok) notifyModelsChanged()
      })
    }, 150)
    return () => clearTimeout(syncTimer)
  }, [backendProcessStatus, isLoaded, notifyModelsChanged, settings])

  const updateSettings = useCallback((patch: Partial<AppSettings> | ((prev: AppSettings) => AppSettings)) => {
    setSettings((prev) => {
      const next = typeof patch === 'function' ? patch(prev) : { ...prev, ...patch }
      // The budget mode and Never split have their own saves. A document edit must not move them.
      const settled =
        next.budgetMode === prev.budgetMode && next.neverSplitIntoParts === prev.neverSplitIntoParts
          ? next
          : { ...next, budgetMode: prev.budgetMode, neverSplitIntoParts: prev.neverSplitIntoParts }
      settingsRef.current = settled
      return settled
    })
  }, [])

  const setBudgetMode = useCallback(async (mode: BudgetMode) => {
    if (budgetModeSavingRef.current) return
    if (settingsRef.current.budgetMode === mode) return
    budgetModeSavingRef.current = true
    setBudgetModeSaving(true)
    setSettingsSyncError(null)
    const result = await ApiClient.updateSettings({ budgetMode: mode })
    budgetModeSavingRef.current = false
    setBudgetModeSaving(false)
    if (!result.ok) {
      setSettingsSyncError(mode === 'cautious' ? CAUTIOUS_ON_SAVE_ERROR : CAUTIOUS_OFF_SAVE_ERROR)
      return
    }
    budgetModeRevisionRef.current += 1
    setSettings((prev) => {
      const next = { ...prev, budgetMode: mode }
      settingsRef.current = next
      return next
    })
  }, [])

  const setNeverSplitIntoParts = useCallback(async (neverSplit: boolean) => {
    if (neverSplitSavingRef.current) return
    if (settingsRef.current.neverSplitIntoParts === neverSplit) return
    neverSplitSavingRef.current = true
    setNeverSplitSaving(true)
    setSettingsSyncError(null)
    const result = await ApiClient.updateSettings({ neverSplitIntoParts: neverSplit })
    neverSplitSavingRef.current = false
    setNeverSplitSaving(false)
    if (!result.ok) {
      setSettingsSyncError(neverSplit ? NEVER_SPLIT_ON_SAVE_ERROR : NEVER_SPLIT_OFF_SAVE_ERROR)
      return
    }
    neverSplitRevisionRef.current += 1
    setSettings((prev) => {
      const next = { ...prev, neverSplitIntoParts: neverSplit }
      settingsRef.current = next
      return next
    })
  }, [])

  const clearSettingsSyncError = useCallback(() => {
    setSettingsSyncError(null)
  }, [])

  const applyAssistantSelection = useCallback((selection: AssistantSelection) => {
    setSettings((prev) =>
      prev.assistantProvider === selection.provider && prev.assistantLocalModel === selection.local_model
        ? prev
        : { ...prev, assistantProvider: selection.provider, assistantLocalModel: selection.local_model },
    )
  }, [])

  // Bumped when a select starts and ends: a read that overlaps a select could return the old choice.
  const assistantSelectionVersion = useRef(0)
  const assistantSelectsPending = useRef(0)
  const [assistantSelectionPending, setAssistantSelectionPending] = useState(false)
  const latestAssistantSelect = useRef(0)

  const selectAssistant = useCallback(async (provider: AppSettings['assistantProvider'], localModel?: AppSettings['assistantLocalModel']) => {
    const id = ++latestAssistantSelect.current
    assistantSelectionVersion.current += 1
    assistantSelectsPending.current += 1
    setAssistantSelectionPending(true)
    try {
      const result = await ApiClient.setAssistantSelection({ provider, local_model: localModel ?? null })
      if (!result.ok) throw new Error(result.error.message)
      if (id === latestAssistantSelect.current) applyAssistantSelection(result.data)
    } finally {
      assistantSelectsPending.current -= 1
      assistantSelectionVersion.current += 1
      setAssistantSelectionPending(assistantSelectsPending.current > 0)
    }
  }, [applyAssistantSelection])

  // Remote can change the shared assistant choice; every Desktop screen reads it from here.
  useEffect(() => {
    if (!isLoaded || backendProcessStatus !== 'alive') return
    const refresh = async () => {
      if (assistantSelectsPending.current > 0) return
      const version = assistantSelectionVersion.current
      const result = await ApiClient.getAssistantSelection()
      if (result.ok && version === assistantSelectionVersion.current) applyAssistantSelection(result.data)
    }
    const timer = setInterval(() => void refresh(), ASSISTANT_SELECTION_REFRESH_MS)
    const onFocus = () => void refresh()
    window.addEventListener('focus', onFocus)
    return () => {
      clearInterval(timer)
      window.removeEventListener('focus', onFocus)
    }
  }, [applyAssistantSelection, backendProcessStatus, isLoaded])

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
    notifyModelsChanged()
  }, [notifyModelsChanged, refreshSettings])

  const saveLtxApiKey = useCallback((value: string) => writeApiKey('ltxApiKey', value), [writeApiKey])
  const saveFalApiKey = useCallback((value: string) => writeApiKey('falApiKey', value), [writeApiKey])
  const saveGeminiApiKey = useCallback((value: string) => writeApiKey('geminiApiKey', value), [writeApiKey])
  const clearLtxApiKey = useCallback(() => writeApiKey('ltxApiKey', ''), [writeApiKey])
  const clearFalApiKey = useCallback(() => writeApiKey('falApiKey', ''), [writeApiKey])
  const clearGeminiApiKey = useCallback(() => writeApiKey('geminiApiKey', ''), [writeApiKey])

  const saveGeminiModel = useCallback(async (geminiModel: string) => {
    const result = await ApiClient.updateSettings({ geminiModel })
    return result.ok
  }, [])

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
      selectAssistant,
      assistantSelectionPending,
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
      setBudgetMode,
      budgetModeSaving,
      setNeverSplitIntoParts,
      neverSplitSaving,
      settingsSyncError,
      clearSettingsSyncError,
      saveGeminiModel,
    }),
    [assistantSelectionPending, budgetModeSaving, clearSettingsSyncError, cudaAvailable, forceApiGenerations, isLoaded, localViable, modelsVersion, neverSplitSaving, notifyModelsChanged, refreshSettings, runtimePolicyLoaded, selectAssistant, clearFalApiKey, clearGeminiApiKey, clearLtxApiKey, saveFalApiKey, saveGeminiApiKey, saveGeminiModel, saveLtxApiKey, setBudgetMode, setNeverSplitIntoParts, settings, settingsSyncError, shouldVideoGenerateWithLtxApi, shouldImageGenerateWithFalApi, updateSettings],
  )

  return (
    <AppSettingsContext.Provider value={contextValue}>
      {settingsSyncError ? (
        <span className="sr-only" role="alert">
          {settingsSyncError}
        </span>
      ) : null}
      {children}
    </AppSettingsContext.Provider>
  )
}

export function useAppSettings() {
  const context = useContext(AppSettingsContext)
  if (!context) {
    throw new Error('useAppSettings must be used within AppSettingsProvider')
  }
  return context
}
