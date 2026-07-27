import { Activity, AlertCircle, Check, Download, Film, Folder, HardDrive, Info, KeyRound, Settings, Sparkles, X, Zap } from 'lucide-react'
import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Button } from './ui/button'
import { BaseModelSection } from './settings/BaseModelSection'
import { useAppSettings, type AppSettings } from '../contexts/AppSettingsContext'
import { ApiClient, type ApiSuccessOf } from '../lib/api-client'
import { logger } from '../lib/logger'
import { ApiKeyHelperRow, LtxApiKeyInput, LtxApiKeyHelperRow } from './LtxApiKeyInput'
import { useHfAuth } from '../hooks/use-hf-auth'
import { useHfModelAccess } from '../hooks/use-hf-model-access'

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  initialTab?: TabId
}

type TabId = 'general' | 'models' | 'apiKeys' | 'promptEnhancer' | 'performance' | 'about'

/** Focuses an API Keys tab input once the modal has switched to that tab.
 *  Shared by the LTX and FAL key inputs — each call gets its own ref/pending state. */
function useApiKeyFocus(isOpen: boolean, activeTab: TabId, setActiveTab: (tab: TabId) => void) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [pending, setPending] = useState(false)

  useEffect(() => {
    if (!isOpen || activeTab !== 'apiKeys' || !pending) return

    const frameId = window.requestAnimationFrame(() => {
      inputRef.current?.focus()
    })
    setPending(false)

    return () => {
      window.cancelAnimationFrame(frameId)
    }
  }, [activeTab, pending, isOpen])

  const openAndFocus = () => {
    setActiveTab('apiKeys')
    setPending(true)
  }

  return { inputRef, openAndFocus }
}

/** A labelled on/off setting row: bolt icon, title, description, switch, and a status pill.
 *  Shared by the CUDA-only Torch Compile + Diffusion Stage Cache toggles. */
function SettingToggle({ title, description, enabled, onToggle, statusOn, statusOff }: {
  title: string
  description: React.ReactNode
  enabled: boolean
  onToggle: () => void
  statusOn: string
  statusOff: string
}) {
  return (
    <div className="space-y-3 pt-4 border-t border-zinc-800">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <svg className="h-4 w-4 text-orange-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
            </svg>
            <label className="text-sm font-medium text-white">{title}</label>
          </div>
          <p className="text-xs text-zinc-500 leading-relaxed">{description}</p>
        </div>

        <button
          type="button"
          onClick={onToggle}
          className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
            enabled ? 'bg-orange-500' : 'bg-zinc-700'
          }`}
        >
          <span
            className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
              enabled ? 'translate-x-5' : 'translate-x-0'
            }`}
          />
        </button>
      </div>

      <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
        enabled ? 'bg-orange-500/10 text-orange-400' : 'bg-zinc-800 text-zinc-500'
      }`}>
        <div className={`w-1.5 h-1.5 rounded-full ${enabled ? 'bg-orange-400' : 'bg-zinc-600'}`} />
        {enabled ? statusOn : statusOff}
      </div>
    </div>
  )
}

export function SettingsModal({ isOpen, onClose, initialTab }: SettingsModalProps) {
  const {
    settings,
    updateSettings,
    saveLtxApiKey,
    saveFalApiKey,
    saveGeminiApiKey,
    forceApiGenerations,
    cudaAvailable,
    mpsAvailable,
    runtimePolicy,
  } = useAppSettings()
  const onSettingsChange = (next: AppSettings) => updateSettings(next)
  const [activeTab, setActiveTab] = useState<TabId>('general')
  const ltxApiKey = useApiKeyFocus(isOpen, activeTab, setActiveTab)
  const falApiKey = useApiKeyFocus(isOpen, activeTab, setActiveTab)
  const [ltxApiKeyInput, setLtxApiKeyInput] = useState('')
  const [falApiKeyInput, setFalApiKeyInput] = useState('')
  const [geminiApiKeyInput, setGeminiApiKeyInput] = useState('')
  const geminiApiKeyInputRef = useRef<HTMLInputElement>(null)
  const [textEncoderRecommendation, setTextEncoderRecommendation] = useState<ApiSuccessOf<'getTextEncoderRecommendation'> | null>(null)
  const [isDownloading, setIsDownloading] = useState(false)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [downloadSessionId, setDownloadSessionId] = useState<string | null>(null)
  const [downloadProgress, setDownloadProgress] = useState<ApiSuccessOf<'getModelDownloadProgress'> | null>(null)
  const { hfAuthStatus, hfAuthPolling, startHuggingFaceLogin, handleHuggingFaceLogout } = useHfAuth(isOpen)
  const textEncoderModelTypes = useMemo(
    () => (forceApiGenerations || !textEncoderRecommendation?.cp_to_download
      ? []
      : [textEncoderRecommendation.cp_to_download]),
    [forceApiGenerations, textEncoderRecommendation?.cp_to_download],
  )
  const { accessMap: teAccessMap, allAuthorized: teAllAuthorized } = useHfModelAccess(textEncoderModelTypes, hfAuthStatus)
  const [appVersion, setAppVersion] = useState('')
  const [noticesText, setNoticesText] = useState<string | null>(null)
  const [noticesLoading, setNoticesLoading] = useState(false)
  const [showNotices, setShowNotices] = useState(false)
  const [modelLicenseText, setModelLicenseText] = useState<string | null>(null)
  const [modelLicenseLoading, setModelLicenseLoading] = useState(false)
  const [showModelLicense, setShowModelLicense] = useState(false)
  const [analyticsEnabled, setAnalyticsEnabled] = useState(false)
  const [projectAssetsPath, setProjectAssetsPath] = useState('')
  const [runtimeTelemetry, setRuntimeTelemetry] = useState<ApiSuccessOf<'getRuntimeTelemetry'> | null>(null)

  // Sync active tab with initialTab prop when modal opens
  useEffect(() => {
    if (isOpen && initialTab) {
      setActiveTab(initialTab)
    }
  }, [isOpen, initialTab])

  // The Models tab is hidden in force-API mode; don't let the selection get stuck there
  // (e.g. via initialTab or a stale value).
  useEffect(() => {
    if (forceApiGenerations && activeTab === 'models') {
      setActiveTab('general')
    }
  }, [forceApiGenerations, activeTab])

  // Fetch app version when About tab is shown
  useEffect(() => {
    if (activeTab !== 'about' || appVersion) return
    window.electronAPI.getAppInfo().then(info => setAppVersion(info.version)).catch(() => {})
  }, [activeTab, appVersion])

  useEffect(() => {
    if (!isOpen || activeTab !== 'performance') return
    let cancelled = false
    const poll = async () => {
      const result = await ApiClient.getRuntimeTelemetry()
      if (result.ok && !cancelled) setRuntimeTelemetry(result.data)
    }
    void poll()
    const interval = window.setInterval(() => { void poll() }, 1000)
    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [activeTab, isOpen])

  // Fetch analytics state when modal opens
  useEffect(() => {
    if (!isOpen) return
    window.electronAPI.getAnalyticsState()
      .then((state: { analyticsEnabled: boolean }) => setAnalyticsEnabled(state.analyticsEnabled))
      .catch(() => {})
    window.electronAPI.getProjectAssetsPath()
      .then((p: string) => setProjectAssetsPath(p))
      .catch(() => {})
  }, [isOpen])

  // Fetch text encoder recommendation when modal opens
  useEffect(() => {
    if (!isOpen || forceApiGenerations) return

    const fetchRecommendation = async () => {
      const result = await ApiClient.getTextEncoderRecommendation()
      if (!result.ok) {
        logger.error(`Failed to fetch text encoder recommendation: ${result.error.message}`)
        return
      }

      const data = result.data
      setTextEncoderRecommendation(data)
      if (data.cp_to_download === null) {
        setIsDownloading(false)
      }
    }

    void fetchRecommendation()
  }, [forceApiGenerations, isOpen])

  // Poll download progress via session ID
  useEffect(() => {
    if (!isDownloading || !downloadSessionId) return

    const poll = async () => {
      const result = await ApiClient.getModelDownloadProgress({ sessionId: downloadSessionId })
      if (!result.ok) return
      setDownloadProgress(result.data)
      if (result.data.status === 'complete') {
        setIsDownloading(false)
        setDownloadSessionId(null)
        const rec = await ApiClient.getTextEncoderRecommendation()
        if (rec.ok) setTextEncoderRecommendation(rec.data)
      } else if (result.data.status === 'error') {
        setDownloadError(result.data.error ?? 'Download failed')
        setIsDownloading(false)
        setDownloadSessionId(null)
      }
    }

    void poll()
    const interval = setInterval(() => { void poll() }, 1000)
    return () => clearInterval(interval)
  }, [isDownloading, downloadSessionId])

  // Handle text encoder download
  const handleDownloadTextEncoder = async () => {
    if (!textEncoderRecommendation?.cp_to_download) return
    setIsDownloading(true)
    setDownloadError(null)
    setDownloadProgress(null)
    const result = await ApiClient.startModelDownload({
      type: 'download',
      cp_ids: [textEncoderRecommendation.cp_to_download],
    })
    if (!result.ok) {
      setDownloadError(result.error.message)
      setIsDownloading(false)
      return
    }
    if (result.data.status === 'started') {
      setDownloadSessionId(result.data.sessionId)
    }
  }

  if (!isOpen) return null

  const handleToggleTorchCompile = () => {
    onSettingsChange({
      ...settings,
      useTorchCompile: !settings.useTorchCompile,
    })
  }

  const handleToggleDiffusionStageCache = () => {
    onSettingsChange({
      ...settings,
      diffusionStageCacheEnabled: !settings.diffusionStageCacheEnabled,
    })
  }

  const handleToggleLocalEncoder = () => {
    onSettingsChange({
      ...settings,
      useLocalTextEncoder: !settings.useLocalTextEncoder,
    })
  }

  const handlePromptCacheSizeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const size = Math.max(0, Math.min(1000, parseInt(e.target.value) || 100))
    onSettingsChange({
      ...settings,
      promptCacheSize: size,
    })
  }

  // Prompt Enhancer handlers
  const handleTogglePromptEnhancer = (mode: 't2v' | 'i2v') => {
    if (mode === 't2v') {
      onSettingsChange({ ...settings, promptEnhancerEnabledT2V: !settings.promptEnhancerEnabledT2V })
    } else {
      onSettingsChange({ ...settings, promptEnhancerEnabledI2V: !settings.promptEnhancerEnabledI2V })
    }
  }
  // Analytics handler
  const handleToggleAnalytics = () => {
    const next = !analyticsEnabled
    setAnalyticsEnabled(next)
    window.electronAPI.setAnalyticsEnabled({ enabled: next }).catch(() => {})
  }

  // Seed handlers
  const handleToggleSeedLock = () => {
    onSettingsChange({
      ...settings,
      seedLocked: !settings.seedLocked,
    })
  }

  const handleLockedSeedChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value) || 0
    onSettingsChange({
      ...settings,
      lockedSeed: Math.max(0, Math.min(2147483647, value)),
    })
  }

  const handleRandomizeSeed = () => {
    onSettingsChange({
      ...settings,
      lockedSeed: Math.floor(Math.random() * 2147483647),
    })
  }

  const handleLoadModelLicense = async () => {
    setModelLicenseLoading(true)
    try {
      const text = await window.electronAPI.fetchLicenseText()
      setModelLicenseText(text)
      setShowModelLicense(true)
    } catch (e) {
      logger.error(`Failed to load model license: ${e}`)
    } finally {
      setModelLicenseLoading(false)
    }
  }

  const handleLoadNotices = async () => {
    setNoticesLoading(true)
    try {
      const text = await window.electronAPI.getNoticesText()
      setNoticesText(text)
      setShowNotices(true)
    } catch (e) {
      logger.error(`Failed to load notices: ${e}`)
    } finally {
      setNoticesLoading(false)
    }
  }

  const tabs = [
    { id: 'general' as TabId, label: 'General', icon: Settings },
    // The Models tab is local-model management — irrelevant (and non-functional) when all
    // generation is forced through the API, so hide it in that mode.
    ...(!forceApiGenerations ? [{ id: 'models' as TabId, label: 'Models', icon: HardDrive }] : []),
    { id: 'apiKeys' as TabId, label: 'API Keys', icon: KeyRound },
    { id: 'promptEnhancer' as TabId, label: 'Prompt Enhancer', icon: Sparkles },
    { id: 'performance' as TabId, label: 'Performance', icon: Activity },
    { id: 'about' as TabId, label: 'About', icon: Info },
  ]

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-2xl mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
          <div className="flex items-center gap-2">
            <Settings className="h-5 w-5 text-zinc-400" />
            <h2 className="text-lg font-semibold text-white">Settings</h2>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="h-8 w-8 text-zinc-400 hover:text-white hover:bg-zinc-800"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-zinc-800">
          {tabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex shrink-0 items-center gap-2 whitespace-nowrap px-4 py-3 text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'text-white border-b-2 border-blue-500 -mb-px'
                    : 'text-zinc-400 hover:text-white'
                }`}
              >
                <Icon className="h-4 w-4" />
                {tab.label}
              </button>
            )
          })}
        </div>

        {/* Content */}
        <div className="px-6 py-5 space-y-6 h-[60vh] overflow-y-auto">
          {activeTab === 'general' && (
            <>
              {/* Project Assets Path */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Download className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">Project Assets Path</h3>
                </div>
                <p className="text-xs text-zinc-500 leading-relaxed">
                  Where generated video and image assets are saved. Each project gets a subfolder.
                </p>
                <div className="flex gap-2">
                  <div className="flex-1 px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-300 text-sm truncate select-text">
                    {projectAssetsPath || <span className="text-zinc-600">Not set</span>}
                  </div>
                  <Button
                    variant="outline"
                    className="border-zinc-700 flex-shrink-0"
                    onClick={async () => {
                      const result = await window.electronAPI.openProjectAssetsPathChangeDialog()
                      if (result.success) {
                        setProjectAssetsPath(result.path)
                      }
                    }}
                  >
                    <Folder className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              {!forceApiGenerations && (
                <div className="space-y-4">
                  <div className="flex items-center gap-2">
                    <Film className="h-4 w-4 text-blue-400" />
                    <h3 className="text-sm font-semibold text-white">Videos Generation</h3>
                  </div>

                  <div
                    className={`bg-zinc-800/50 rounded-lg p-4 border-2 transition-colors cursor-pointer ${
                      settings.userPrefersLtxApiVideoGenerations ? 'border-blue-500' : 'border-transparent hover:border-zinc-600'
                    }`}
                    onClick={() => {
                      if (!settings.hasLtxApiKey) {
                        ltxApiKey.openAndFocus()
                        return
                      }
                      onSettingsChange({
                        ...settings,
                        userPrefersLtxApiVideoGenerations: !settings.userPrefersLtxApiVideoGenerations,
                      })
                    }}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <Zap className="h-4 w-4 text-blue-400" />
                          <span className="text-sm font-medium text-white">Use LTX API Models</span>
                        </div>
                        <p className="text-xs text-zinc-400 mt-1">
                          Unlocks LTX-2.3 Fast and Pro in the model selector. Local generation supports Fast only.
                        </p>
                      </div>
                      <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                        settings.userPrefersLtxApiVideoGenerations ? 'border-blue-500 bg-blue-500' : 'border-zinc-600'
                      }`}>
                        {settings.userPrefersLtxApiVideoGenerations && <Check className="h-3 w-3 text-white" />}
                      </div>
                    </div>

                    {!settings.hasLtxApiKey && (
                      <div className="mt-2 text-xs text-amber-400 flex items-center gap-1.5">
                        <AlertCircle className="h-3 w-3" />
                        API key required — configure it in the API Keys tab.
                      </div>
                    )}
                  </div>
                </div>
              )}

              {!forceApiGenerations && (
                <div className="space-y-4">
                  <div className="flex items-center gap-2">
                    <Sparkles className="h-4 w-4 text-blue-400" />
                    <h3 className="text-sm font-semibold text-white">Images Generation</h3>
                  </div>

                  <div
                    className={`bg-zinc-800/50 rounded-lg p-4 border-2 transition-colors cursor-pointer ${
                      settings.userPrefersFalApiImageGenerations ? 'border-blue-500' : 'border-transparent hover:border-zinc-600'
                    }`}
                    onClick={() => {
                      if (!settings.hasFalApiKey) {
                        falApiKey.openAndFocus()
                        return
                      }
                      onSettingsChange({
                        ...settings,
                        userPrefersFalApiImageGenerations: !settings.userPrefersFalApiImageGenerations,
                      })
                    }}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <Zap className="h-4 w-4 text-blue-400" />
                          <span className="text-sm font-medium text-white">Generate With API</span>
                        </div>
                        <p className="text-xs text-zinc-400 mt-1">
                          Use the FAL API for image generation and editing when a FAL API key is configured.
                        </p>
                      </div>
                      <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                        settings.userPrefersFalApiImageGenerations ? 'border-blue-500 bg-blue-500' : 'border-zinc-600'
                      }`}>
                        {settings.userPrefersFalApiImageGenerations && <Check className="h-3 w-3 text-white" />}
                      </div>
                    </div>

                    {!settings.hasFalApiKey && (
                      <div className="mt-2 text-xs text-amber-400 flex items-center gap-1.5">
                        <AlertCircle className="h-3 w-3" />
                        API key required — configure it in the API Keys tab.
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Text Encoding Section */}
              {!forceApiGenerations && (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <svg className="h-4 w-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M15 7h3a5 5 0 0 1 5 5 5 5 0 0 1-5 5h-3m-6 0H6a5 5 0 0 1-5-5 5 5 0 0 1 5-5h3" />
                    <line x1="8" y1="12" x2="16" y2="12" />
                  </svg>
                  <h3 className="text-sm font-semibold text-white">Text Encoding</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Text encoding converts your prompt into data the AI understands. Choose how to do this.
                </p>

                {/* LTX API Option (Default) */}
                <div
                  className={`bg-zinc-800/50 rounded-lg p-4 border-2 transition-colors cursor-pointer ${
                    !settings.useLocalTextEncoder ? 'border-blue-500' : 'border-transparent hover:border-zinc-600'
                  }`}
                  onClick={() => {
                    if (!settings.useLocalTextEncoder) return
                    if (!settings.hasLtxApiKey) {
                      ltxApiKey.openAndFocus()
                      return
                    }
                    handleToggleLocalEncoder()
                  }}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <Zap className="h-4 w-4 text-blue-400" />
                        <span className="text-sm font-medium text-white">LTX API</span>
                        <span className="text-xs px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded">Recommended</span>
                      </div>
                      <p className="text-xs text-zinc-400 mt-1">
                        Fast cloud-based text encoding (~1 second). Requires an LTX API key configured in the API Keys tab.
                      </p>
                    </div>
                    <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                      !settings.useLocalTextEncoder ? 'border-blue-500 bg-blue-500' : 'border-zinc-600'
                    }`}>
                      {!settings.useLocalTextEncoder && <Check className="h-3 w-3 text-white" />}
                    </div>
                  </div>

                  {/* Warning when selected but no key */}
                  {!settings.useLocalTextEncoder && !settings.hasLtxApiKey && (
                    <div className="mt-2 text-xs text-amber-400 flex items-center gap-1.5">
                      <AlertCircle className="h-3 w-3" />
                      API key required — configure it in the API Keys tab.
                    </div>
                  )}

                  {/* Prompt Cache Size — only relevant for API text encoding */}
                  {!settings.useLocalTextEncoder && settings.hasLtxApiKey && (
                    <div className="flex items-center justify-between mt-3 pt-3 border-t border-zinc-700/50">
                      <div>
                        <label className="text-xs text-white">Prompt Cache</label>
                        <p className="text-xs text-zinc-500">Skip repeat encoding calls</p>
                      </div>
                      <input
                        type="number"
                        min="0"
                        max="1000"
                        value={settings.promptCacheSize ?? 100}
                        onChange={handlePromptCacheSizeChange}
                        onClick={(e) => e.stopPropagation()}
                        className="w-16 px-2 py-1 bg-zinc-700 border border-zinc-600 rounded text-xs text-white text-center focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                  )}
                </div>

                {/* Local Encoder Option */}
                <div
                  className={`bg-zinc-800/50 rounded-lg p-4 border-2 transition-colors cursor-pointer ${
                    settings.useLocalTextEncoder ? 'border-blue-500' : 'border-transparent hover:border-zinc-600'
                  }`}
                  onClick={() => !settings.useLocalTextEncoder && handleToggleLocalEncoder()}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <svg className="h-4 w-4 text-zinc-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <rect x="4" y="4" width="16" height="16" rx="2" />
                          <path d="M9 9h6m-6 3h6m-6 3h4" />
                        </svg>
                        <span className="text-sm font-medium text-white">Local Encoder</span>
                      </div>
                      <p className="text-xs text-zinc-400 mt-1">
                        Run on your computer (~23 seconds). Requires 25 GB download.
                      </p>
                    </div>
                    <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                      settings.useLocalTextEncoder ? 'border-blue-500 bg-blue-500' : 'border-zinc-600'
                    }`}>
                      {settings.useLocalTextEncoder && <Check className="h-3 w-3 text-white" />}
                    </div>
                  </div>

                  {/* Download Status - show when this option is selected */}
                  {settings.useLocalTextEncoder && (
                    <div className="mt-3 pt-3 border-t border-zinc-700/50">
                      {textEncoderRecommendation?.cp_to_download === null ? (
                        <div className="flex items-center gap-2 text-xs text-green-400">
                          <Check className="h-4 w-4" />
                          <span>Downloaded ({textEncoderRecommendation?.expected_size_gb ?? 0} GB)</span>
                        </div>
                      ) : isDownloading ? (
                        <div className="space-y-1.5">
                          <div className="flex items-center justify-between text-[11px]">
                            <span className="text-zinc-300">Downloading text encoder...</span>
                            <span className="text-zinc-500">{downloadProgress?.status === 'downloading' ? Math.round(downloadProgress.current_file_progress) : 0}%</span>
                          </div>
                          <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                            <div className="h-full transition-all duration-300 bg-blue-500" style={{ width: `${downloadProgress?.status === 'downloading' ? downloadProgress.current_file_progress : 0}%` }} />
                          </div>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2 text-xs text-amber-400">
                            <AlertCircle className="h-4 w-4" />
                            <span>Not downloaded ({textEncoderRecommendation?.expected_size_gb || 0} GB required)</span>
                          </div>
                          {hfAuthStatus === 'authenticated' && !teAllAuthorized && Object.keys(teAccessMap).length > 0 && (
                            <div className="space-y-1.5 mb-2">
                              {Object.entries(teAccessMap)
                                .filter(([, status]) => status === 'not_authorized')
                                .map(([repoId]) => (
                                  <div key={repoId} className="flex items-center justify-between bg-zinc-900 rounded px-2 py-1.5">
                                    <span className="text-[10px] text-zinc-400 font-mono">{repoId}</span>
                                    <button
                                      onClick={(e) => { e.stopPropagation(); window.electronAPI.openHuggingFaceRepo({ repoId }) }}
                                      className="text-[10px] text-indigo-400 hover:text-indigo-300 font-medium"
                                    >
                                      Request access
                                    </button>
                                  </div>
                                ))}
                            </div>
                          )}
                          <Button
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation()
                              void handleDownloadTextEncoder()
                            }}
                            disabled={!textEncoderRecommendation?.cp_to_download || !teAllAuthorized}
                            className="w-full bg-blue-600 hover:bg-blue-500 text-white text-xs"
                          >
                            <Download className="h-3 w-3 mr-2" />
                            Download Text Encoder
                          </Button>
                          {downloadError && (
                            <p className="text-xs text-red-400">{downloadError}</p>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
              )}

              {/* torch.compile remains CUDA-only. */}
              {cudaAvailable && (
                <SettingToggle
                  title="Torch Compile"
                  description={<>Compiles the model for optimized inference. <span className="text-orange-400">Experimental:</span> First
                    generation can take 5-10+ minutes for compilation. Subsequent generations may be
                    20-40% faster. Requires app restart to take effect.</>}
                  enabled={settings.useTorchCompile}
                  onToggle={handleToggleTorchCompile}
                  statusOn="Optimized inference (recommended)"
                  statusOff="Standard inference"
                />
              )}

              {(cudaAvailable || mpsAvailable) && (
                <SettingToggle
                  title="Diffusion Stage Cache"
                  description={<>Reuses an already-built transformer across stage 1/stage 2 within one generation
                    instead of reloading it from disk twice, then releases it before VAE decode. <span className="text-orange-400">Experimental:</span> applies to the official Torch runtime; MLX manages stages independently.</>}
                  enabled={settings.diffusionStageCacheEnabled}
                  onToggle={handleToggleDiffusionStageCache}
                  statusOn="Skipping redundant transformer reloads"
                  statusOff="Standard behavior"
                />
              )}

              {/* Seed Lock Setting */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <svg className="h-4 w-4 text-emerald-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                        <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                      </svg>
                      <label className="text-sm font-medium text-white">
                        Lock Seed
                      </label>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Use the same seed for reproducible generations. When unlocked, a random seed is used each time.
                    </p>
                  </div>

                  {/* Toggle Switch */}
                  <button
                    onClick={handleToggleSeedLock}
                    className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      settings.seedLocked ? 'bg-emerald-500' : 'bg-zinc-700'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        settings.seedLocked ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>

                {/* Seed input - only show when locked */}
                {settings.seedLocked && (
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      min="0"
                      max="2147483647"
                      value={settings.lockedSeed ?? 42}
                      onChange={handleLockedSeedChange}
                      className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                      placeholder="Enter seed..."
                    />
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleRandomizeSeed}
                      className="h-9 px-3 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800"
                      title="Generate random seed"
                    >
                      <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16" />
                      </svg>
                    </Button>
                  </div>
                )}

                {/* Status indicator */}
                <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                  settings.seedLocked
                    ? 'bg-emerald-500/10 text-emerald-400'
                    : 'bg-zinc-800 text-zinc-500'
                }`}>
                  <div className={`w-1.5 h-1.5 rounded-full ${
                    settings.seedLocked ? 'bg-emerald-400' : 'bg-zinc-600'
                  }`} />
                  {settings.seedLocked ? `Seed locked: ${settings.lockedSeed ?? 42}` : 'Random seed each generation'}
                </div>
              </div>

              {/* Anonymous Analytics Setting */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <svg className="h-4 w-4 text-violet-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="20" x2="18" y2="10" />
                        <line x1="12" y1="20" x2="12" y2="4" />
                        <line x1="6" y1="20" x2="6" y2="14" />
                      </svg>
                      <label className="text-sm font-medium text-white">
                        Anonymous Analytics
                      </label>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Share anonymous usage data to help improve LTX Desktop.
                      Only basic technical information is collected — never personal data or generated content.
                    </p>
                  </div>

                  {/* Toggle Switch */}
                  <button
                    onClick={handleToggleAnalytics}
                    className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      analyticsEnabled ? 'bg-violet-500' : 'bg-zinc-700'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        analyticsEnabled ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>

              </div>
            </>
          )}

          {activeTab === 'models' && !forceApiGenerations && <BaseModelSection />}

          {activeTab === 'apiKeys' && (
            <>
              {/* LTX API Key Section */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Zap className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">LTX API</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Your LTX API key is used for cloud text encoding, prompt enhancement, and API video generation.
                  Add your key below to unlock these features.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex gap-2">
                    <LtxApiKeyInput
                      ref={ltxApiKey.inputRef}
                      value={ltxApiKeyInput}
                      onChange={(e) => setLtxApiKeyInput(e.target.value)}
                      placeholder={settings.hasLtxApiKey ? 'Enter new key to replace...' : 'Enter your LTX API key...'}
                      stopPropagation
                      className="flex-1"
                    />
                    <button
                      onClick={() => {
                        const trimmed = ltxApiKeyInput.trim()
                        if (!trimmed) return
                        void saveLtxApiKey(trimmed)
                        setLtxApiKeyInput('')
                      }}
                      disabled={!ltxApiKeyInput.trim()}
                      className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    >
                      Save Key
                    </button>
                  </div>
                  <LtxApiKeyHelperRow stopPropagation />
                  <div className="flex items-center justify-between">
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                      settings.hasLtxApiKey
                        ? 'bg-green-500/10 text-green-400'
                        : 'bg-amber-500/10 text-amber-400'
                    }`}>
                      {settings.hasLtxApiKey ? (
                        <>
                          <Check className="h-3 w-3" />
                          Key configured
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-3 w-3" />
                          API key required
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* FAL API Key Section */}
              <div className="space-y-4 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <KeyRound className="h-4 w-4 text-cyan-400" />
                  <h3 className="text-sm font-semibold text-white">FAL AI</h3>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">Optional</span>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Your FAL AI key is used for generating or editing images with Z Image Turbo when API generations are enabled.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex gap-2">
                    <LtxApiKeyInput
                      ref={falApiKey.inputRef}
                      value={falApiKeyInput}
                      onChange={(e) => setFalApiKeyInput(e.target.value)}
                      placeholder={settings.hasFalApiKey ? 'Enter new key to replace...' : 'Enter your FAL AI API key...'}
                      stopPropagation
                      className="flex-1"
                    />
                    <button
                      onClick={() => {
                        const trimmed = falApiKeyInput.trim()
                        if (!trimmed) return
                        void saveFalApiKey(trimmed)
                        setFalApiKeyInput('')
                      }}
                      disabled={!falApiKeyInput.trim()}
                      className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    >
                      Save Key
                    </button>
                  </div>
                  <ApiKeyHelperRow
                    stopPropagation
                    label="Get FAL API key"
                    onOpenKey={() => window.electronAPI.openFalApiKeyPage()}
                  />
                  <div className="flex items-center justify-between">
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                      settings.hasFalApiKey
                        ? 'bg-green-500/10 text-green-400'
                        : 'bg-zinc-800 text-zinc-500'
                    }`}>
                      {settings.hasFalApiKey ? (
                        <>
                          <Check className="h-3 w-3" />
                          Key configured
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-3 w-3" />
                          Optional
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Gemini API Key Section */}
              <div className="space-y-4 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-purple-400" />
                  <h3 className="text-sm font-semibold text-white">Gemini API</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Your Gemini API key is used for AI-powered prompt suggestions when filling timeline gaps, and for the Enhance (API) prompt enhancer.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex gap-2">
                    <input
                      ref={geminiApiKeyInputRef}
                      type="password"
                      value={geminiApiKeyInput}
                      onChange={(e) => setGeminiApiKeyInput(e.target.value)}
                      placeholder={settings.hasGeminiApiKey ? 'Enter new key to replace...' : 'Enter your Gemini API key...'}
                      onKeyDown={(e) => e.stopPropagation()}
                      className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <button
                      onClick={() => {
                        const trimmed = geminiApiKeyInput.trim()
                        if (!trimmed) return
                        void saveGeminiApiKey(trimmed)
                        setGeminiApiKeyInput('')
                      }}
                      disabled={!geminiApiKeyInput.trim()}
                      className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    >
                      Save Key
                    </button>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                      settings.hasGeminiApiKey
                        ? 'bg-green-500/10 text-green-400'
                        : 'bg-amber-500/10 text-amber-400'
                    }`}>
                      {settings.hasGeminiApiKey ? (
                        <>
                          <Check className="h-3 w-3" />
                          Key configured
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-3 w-3" />
                          API key required
                        </>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <a
                      href="https://aistudio.google.com/app/apikey"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-400 hover:text-blue-300 transition-colors underline underline-offset-2"
                      onClick={(e) => e.stopPropagation()}
                    >
                      Get Gemini API key →
                    </a>
                  </div>
                </div>
              </div>

              {/* HuggingFace Account */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Download className="h-4 w-4 text-orange-400" />
                  <h3 className="text-sm font-semibold text-white">HuggingFace</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Sign in to HuggingFace to download model files.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                    hfAuthStatus === 'authenticated'
                      ? 'bg-green-500/10 text-green-400'
                      : 'bg-amber-500/10 text-amber-400'
                  }`}>
                    {hfAuthStatus === 'authenticated' ? (
                      <>
                        <Check className="h-3 w-3" />
                        Signed in
                      </>
                    ) : (
                      <>
                        <AlertCircle className="h-3 w-3" />
                        Not signed in
                      </>
                    )}
                  </div>

                  {hfAuthStatus === 'authenticated' ? (
                    <button
                      onClick={handleHuggingFaceLogout}
                      className="px-3 py-2 bg-zinc-700 text-white text-sm rounded-lg hover:bg-zinc-600 transition-colors"
                    >
                      Sign out
                    </button>
                  ) : (
                    <button
                      onClick={startHuggingFaceLogin}
                      disabled={hfAuthPolling}
                      className="px-3 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors"
                    >
                      {hfAuthPolling ? 'Waiting for sign in...' : 'Sign in with HuggingFace'}
                    </button>
                  )}
                </div>
              </div>
            </>
          )}

          {activeTab === 'promptEnhancer' && (
            <>
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">Prompt Enhancer</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Automatically enhances your prompts via the LTX API with rich visual details, sound descriptions,
                  and motion cues to help generate higher quality videos. Control independently for each generation type.
                </p>

                {!settings.hasLtxApiKey ? (
                  <div className="space-y-4 mt-2">
                    <div className="bg-amber-500/5 border border-amber-500/20 rounded-lg p-4 space-y-3">
                      <div className="flex items-start gap-2.5">
                        <AlertCircle className="h-4 w-4 text-amber-400 mt-0.5 flex-shrink-0" />
                        <div className="space-y-2">
                          <p className="text-sm text-amber-300 font-medium">LTX API key required</p>
                          <p className="text-xs text-zinc-400 leading-relaxed">
                            Prompt enhancement runs server-side on the LTX API. To use this feature, you need to configure
                            an API key in the API Keys tab.
                          </p>
                        </div>
                      </div>
                      <button
                        onClick={() => setActiveTab('apiKeys')}
                        className="w-full mt-1 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-lg transition-colors"
                      >
                        Set API Key
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    {/* T2V Toggle */}
                    <div
                      className="flex items-center justify-between bg-zinc-800/50 rounded-lg px-4 py-3 border border-zinc-700/50 cursor-pointer"
                      onClick={() => handleTogglePromptEnhancer('t2v')}
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-xs font-semibold text-blue-400 bg-blue-400/10 px-1.5 py-0.5 rounded">T2V</span>
                        <div>
                          <span className="text-sm text-zinc-200">Text-to-Video</span>
                          <p className="text-[10px] text-zinc-500 mt-0.5">
                            {settings.promptEnhancerEnabledT2V ? 'Prompts will be enhanced before T2V generation' : 'T2V prompts used as-is'}
                          </p>
                        </div>
                      </div>
                      <div className={`relative w-11 h-6 rounded-full transition-colors flex-shrink-0 ${
                        settings.promptEnhancerEnabledT2V ? 'bg-blue-500' : 'bg-zinc-700'
                      }`}>
                        <div className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform pointer-events-none ${
                          settings.promptEnhancerEnabledT2V ? 'translate-x-5' : 'translate-x-0'
                        }`} />
                      </div>
                    </div>

                    {/* I2V Toggle */}
                    <div
                      className="flex items-center justify-between bg-zinc-800/50 rounded-lg px-4 py-3 border border-zinc-700/50 cursor-pointer"
                      onClick={() => handleTogglePromptEnhancer('i2v')}
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-xs font-semibold text-emerald-400 bg-emerald-400/10 px-1.5 py-0.5 rounded">I2V</span>
                        <div>
                          <span className="text-sm text-zinc-200">Image-to-Video</span>
                          <p className="text-[10px] text-zinc-500 mt-0.5">
                            {settings.promptEnhancerEnabledI2V ? 'Prompts will be enhanced before I2V generation' : 'I2V prompts used as-is'}
                          </p>
                        </div>
                      </div>
                      <div className={`relative w-11 h-6 rounded-full transition-colors flex-shrink-0 ${
                        settings.promptEnhancerEnabledI2V ? 'bg-blue-500' : 'bg-zinc-700'
                      }`}>
                        <div className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform pointer-events-none ${
                          settings.promptEnhancerEnabledI2V ? 'translate-x-5' : 'translate-x-0'
                        }`} />
                      </div>
                    </div>
                  </>
                )}
              </div>
            </>
          )}

          {activeTab === 'performance' && (
            <div className="space-y-5">
              <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3 border border-zinc-700/50">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold text-white">Local runtime policy</h3>
                    <p className="text-xs text-zinc-500 mt-1">
                      Capability-aware selection keeps feature-rich requests on Torch and uses MLX only where parity is qualified.
                    </p>
                  </div>
                  <span className="text-xs uppercase tracking-wide px-2 py-1 rounded bg-blue-500/10 text-blue-300">
                    {runtimePolicy?.auto_fast_video_engine ?? '...'}
                  </span>
                </div>
                {runtimePolicy && (
                  <>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="bg-zinc-900/60 rounded p-3">
                        <div className="text-zinc-500">Preference</div>
                        <div className="text-zinc-200 mt-1">{runtimePolicy.fast_video_engine_preference}</div>
                      </div>
                      <div className="bg-zinc-900/60 rounded p-3">
                        <div className="text-zinc-500">Memory mode</div>
                        <div className="text-zinc-200 mt-1">
                          {runtimePolicy.execution_mode}{runtimePolicy.automatic_tiling ? ' + auto tiling' : ''}
                        </div>
                      </div>
                    </div>
                    <p className="text-xs text-zinc-400">{runtimePolicy.auto_selection_reason}</p>
                    <div className="text-xs text-zinc-500 break-all">
                      Model: <span className="text-zinc-300">{runtimePolicy.mlx_model_source}</span> ({runtimePolicy.mlx_model_variant})
                    </div>
                    {runtimePolicy.quality_warning && (
                      <div className="text-xs text-amber-300 bg-amber-500/10 rounded p-3">
                        {runtimePolicy.quality_warning}
                      </div>
                    )}
                  </>
                )}
              </div>

              <div className="space-y-2">
                <h3 className="text-sm font-semibold text-white">Live / last-job memory</h3>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {[
                    ['Process RSS', runtimeTelemetry?.process_rss_mib],
                    ['System available', runtimeTelemetry?.system_available_mib],
                    ['MLX active', runtimeTelemetry?.mlx_active_mib],
                    ['MLX cache', runtimeTelemetry?.mlx_cache_mib],
                    ['MLX peak', runtimeTelemetry?.mlx_peak_mib],
                    ['MPS allocated', runtimeTelemetry?.mps_allocated_mib],
                    ['MPS driver', runtimeTelemetry?.mps_driver_mib],
                  ].map(([label, value]) => (
                    <div key={String(label)} className="bg-zinc-800/50 rounded p-3 border border-zinc-700/40">
                      <div className="text-zinc-500">{label}</div>
                      <div className="text-zinc-200 mt-1">
                        {typeof value === 'number' ? `${value.toLocaleString()} MiB` : '—'}
                      </div>
                    </div>
                  ))}
                </div>
                {runtimeTelemetry?.mlx_profile_status && (
                  <div className="bg-zinc-800/50 rounded p-3 border border-zinc-700/40 text-xs space-y-1">
                    <div className="flex justify-between gap-3">
                      <span className="text-zinc-500">MLX profile</span>
                      <span className="text-zinc-200">{runtimeTelemetry.mlx_profile_status}</span>
                    </div>
                    <div className="text-zinc-400">
                      Phase: {runtimeTelemetry.mlx_profile_phase ?? 'initializing'}
                    </div>
                    {runtimeTelemetry.mlx_runtime_identity && (
                      <div className="text-zinc-400">
                        Runtime {String(runtimeTelemetry.mlx_runtime_identity.runtime_version ?? 'unknown')}
                        {' · '}{String(runtimeTelemetry.mlx_runtime_identity.runtime_commit ?? 'unknown').slice(0, 12)}
                        {' · '}{String(runtimeTelemetry.mlx_runtime_identity.device_name ?? 'unknown device')}
                      </div>
                    )}
                    {runtimeTelemetry.mlx_profile_path && (
                      <div className="text-zinc-600 break-all">{runtimeTelemetry.mlx_profile_path}</div>
                    )}
                  </div>
                )}
              </div>

              <div className="bg-zinc-800/50 rounded-lg p-4 space-y-2 border border-zinc-700/50">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-white">Shared local accelerator</h3>
                  <span className={`text-xs px-2 py-1 rounded ${
                    runtimeTelemetry?.local_metal_lease_status === 'waiting'
                      ? 'bg-amber-500/10 text-amber-300'
                      : runtimeTelemetry?.local_metal_lease_status === 'held'
                        ? 'bg-emerald-500/10 text-emerald-300'
                        : 'bg-zinc-700 text-zinc-400'
                  }`}>
                    {runtimeTelemetry?.local_metal_lease_status ?? 'idle'}
                  </span>
                </div>
                <p className="text-xs text-zinc-500">
                  LTX apps coordinate through one advisory Metal lease so two heavy local renders cannot exhaust unified memory.
                </p>
                {runtimeTelemetry?.local_metal_lease_reason && (
                  <p className="text-xs text-zinc-300">{runtimeTelemetry.local_metal_lease_reason}</p>
                )}
                {runtimeTelemetry?.local_metal_lease_owner && (
                  <p className="text-xs text-zinc-400">
                    Owner: {String(runtimeTelemetry.local_metal_lease_owner.product ?? 'unknown')}
                    {' '}· PID {String(runtimeTelemetry.local_metal_lease_owner.pid ?? '?')}
                    {' '}· {runtimeTelemetry.local_metal_lease_waited_seconds.toFixed(1)}s
                  </p>
                )}
              </div>

              {runtimePolicy && (
                <div className="space-y-3">
                  <h3 className="text-sm font-semibold text-white">Capability routing</h3>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                    {Object.entries(runtimePolicy.capability_engines).map(([capability, engine]) => (
                      <div key={capability} className="flex justify-between gap-2 border-b border-zinc-800 pb-1">
                        <span className="text-zinc-500">{capability.replace(/_/g, ' ')}</span>
                        <span className="text-zinc-300 uppercase">{engine}</span>
                      </div>
                    ))}
                  </div>
                  <h3 className="text-sm font-semibold text-white pt-2">Runtime provenance</h3>
                  {runtimePolicy.provenance.map(item => (
                    <div key={item.component} className="text-xs bg-zinc-800/40 rounded p-3">
                      <div className="flex justify-between gap-3">
                        <span className="text-zinc-200">{item.component}</span>
                        <span className="text-zinc-400">{item.version}</span>
                      </div>
                      {item.revision && <div className="text-zinc-600 font-mono mt-1 truncate">{item.revision}</div>}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'about' && (
            <>
              {showModelLicense ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-white">LTX-2 Model License</h3>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowModelLicense(false)}
                      className="h-7 px-2 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800"
                    >
                      Back
                    </Button>
                  </div>
                  <pre className="text-xs text-zinc-300 whitespace-pre-wrap font-mono bg-zinc-800/50 rounded-lg p-4 max-h-[50vh] overflow-y-auto border border-zinc-700/50">
                    {modelLicenseText}
                  </pre>
                </div>
              ) : showNotices ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-white">Third-Party Notices</h3>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowNotices(false)}
                      className="h-7 px-2 text-xs text-zinc-400 hover:text-white hover:bg-zinc-800"
                    >
                      Back
                    </Button>
                  </div>
                  <pre className="text-xs text-zinc-300 whitespace-pre-wrap font-mono bg-zinc-800/50 rounded-lg p-4 max-h-[50vh] overflow-y-auto border border-zinc-700/50">
                    {noticesText}
                  </pre>
                </div>
              ) : (
                <div className="space-y-6">
                  {/* App Identity */}
                  <div className="text-center space-y-2">
                    <h3 className="text-lg font-bold text-white">LTX Desktop</h3>
                    <p className="text-sm text-zinc-400">Version {appVersion || '...'}</p>
                    <p className="text-xs text-zinc-500">AI-Powered Video Editor</p>
                  </div>

                  {/* License */}
                  <div className="bg-zinc-800/50 rounded-lg p-4 space-y-2">
                    <div className="flex items-center gap-2">
                      <Info className="h-4 w-4 text-blue-400" />
                      <span className="text-sm font-medium text-white">License</span>
                    </div>
                    <p className="text-xs text-zinc-400">
                      Licensed under the Apache License, Version 2.0
                    </p>
                  </div>

                  {/* LTX-2 Model License */}
                  <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                    <div className="flex items-center gap-2">
                      <svg className="h-4 w-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                      </svg>
                      <span className="text-sm font-medium text-white">LTX-2 Model License</span>
                    </div>
                    <p className="text-xs text-zinc-400">
                      The LTX-2 model is subject to the LTX-2 Community License Agreement, accepted during first-run setup.
                    </p>
                    <Button
                      size="sm"
                      onClick={handleLoadModelLicense}
                      disabled={modelLicenseLoading}
                      className="w-full bg-zinc-700 hover:bg-zinc-600 text-white text-xs"
                    >
                      {modelLicenseLoading ? 'Loading...' : 'View Model License'}
                    </Button>
                  </div>

                  {/* Third-Party Notices */}
                  <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                    <div className="flex items-center gap-2">
                      <svg className="h-4 w-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                        <line x1="16" y1="13" x2="8" y2="13" />
                        <line x1="16" y1="17" x2="8" y2="17" />
                      </svg>
                      <span className="text-sm font-medium text-white">Third-Party Notices</span>
                    </div>
                    <p className="text-xs text-zinc-400">
                      This application uses open-source software and AI models subject to their own license terms.
                    </p>
                    <Button
                      size="sm"
                      onClick={handleLoadNotices}
                      disabled={noticesLoading}
                      className="w-full bg-zinc-700 hover:bg-zinc-600 text-white text-xs"
                    >
                      {noticesLoading ? 'Loading...' : 'View Third-Party Notices'}
                    </Button>
                  </div>

                  {/* Copyright */}
                  <p className="text-center text-xs text-zinc-600">
                    Copyright © 2026 Lightricks
                  </p>
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-zinc-800 flex justify-end">
          <Button
            onClick={onClose}
            className="bg-zinc-700 hover:bg-zinc-600 text-white"
          >
            Done
          </Button>
        </div>
      </div>
    </div>
  )
}

export type { AppSettings, TabId as SettingsTabId }
