import { AlertCircle, Check, Cpu, Download, Film, FolderOpen, Info, KeyRound, RefreshCw, Settings, Sliders, Sparkles, X, Zap } from 'lucide-react'
import React, { useEffect, useRef, useState } from 'react'
import { Button } from './ui/button'
import { ModelGuideDialog } from './ModelGuideDialog'
import { useAppSettings, type AppSettings } from '../contexts/AppSettingsContext'
import { logger } from '../lib/logger'
import { ApiKeyHelperRow, LtxApiKeyInput, LtxApiKeyHelperRow } from './LtxApiKeyInput'

interface TextEncoderStatus {
  downloaded: boolean
  size_gb: number
  expected_size_gb: number
}

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  initialTab?: TabId
}

type TabId = 'general' | 'apiKeys' | 'inference' | 'promptEnhancer' | 'models' | 'about'

export function SettingsModal({ isOpen, onClose, initialTab }: SettingsModalProps) {
  const { settings, updateSettings, saveLtxApiKey, saveReplicateApiKey, saveGeminiApiKey, saveCivitaiApiKey, refreshSettings, forceApiGenerations } = useAppSettings()
  const onSettingsChange = (next: AppSettings) => updateSettings(next)
  const [activeTab, setActiveTab] = useState<TabId>('general')
  const [ltxApiKeyInput, setLtxApiKeyInput] = useState('')
  const ltxApiKeyInputRef = useRef<HTMLInputElement>(null)
  const [focusLtxApiKeyInputOnTabChange, setFocusLtxApiKeyInputOnTabChange] = useState(false)
  const [replicateApiKeyInput, setReplicateApiKeyInput] = useState('')
  const replicateApiKeyInputRef = useRef<HTMLInputElement>(null)
  const [geminiApiKeyInput, setGeminiApiKeyInput] = useState('')
  const geminiApiKeyInputRef = useRef<HTMLInputElement>(null)
  const [paletteApiKeyInput, setPaletteApiKeyInput] = useState('')
  const [paletteStatus, setPaletteStatus] = useState<{ connected: boolean; user: { email: string; name: string } | null; error?: string } | null>(null)
  const [paletteCredits, setPaletteCredits] = useState<number | null>(null)
  const [paletteLoginEmail, setPaletteLoginEmail] = useState('')
  const [paletteLoginPassword, setPaletteLoginPassword] = useState('')
  const [paletteLoginError, setPaletteLoginError] = useState<string | null>(null)
  const [paletteLoginLoading, setPaletteLoginLoading] = useState(false)
  const [paletteAuthMode, setPaletteAuthMode] = useState<'login' | 'apikey'>('login')
  const [loraSyncing, setLoraSyncing] = useState(false)
  const [loraSyncResult, setLoraSyncResult] = useState<string | null>(null)
  const [civitaiApiKeyInput, setCivitaiApiKeyInput] = useState('')
  const [textEncoderStatus, setTextEncoderStatus] = useState<TextEncoderStatus | null>(null)
  const [isDownloading, setIsDownloading] = useState(false)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [appVersion, setAppVersion] = useState('')
  const [noticesText, setNoticesText] = useState<string | null>(null)
  const [noticesLoading, setNoticesLoading] = useState(false)
  const [showNotices, setShowNotices] = useState(false)
  const [modelLicenseText, setModelLicenseText] = useState<string | null>(null)
  const [modelLicenseLoading, setModelLicenseLoading] = useState(false)
  const [showModelLicense, setShowModelLicense] = useState(false)
  const [analyticsEnabled, setAnalyticsEnabled] = useState(false)
  const [videoModels, setVideoModels] = useState<any[]>([])
  const [distilledLoraFound, setDistilledLoraFound] = useState(false)
  const [modelScanning, setModelScanning] = useState(false)
  const [gpuInfo, setGpuInfo] = useState<{ name: string | null; vram: number | null } | null>(null)
  const [showModelGuide, setShowModelGuide] = useState(false)

  // Sync active tab with initialTab prop when modal opens
  useEffect(() => {
    if (isOpen && initialTab) {
      setActiveTab(initialTab)
    }
  }, [isOpen, initialTab])

  // Poll Palette sync status when on apiKeys tab and key is set
  useEffect(() => {
    if (!isOpen || activeTab !== 'apiKeys' || !settings.hasPaletteApiKey) {
      setPaletteStatus(null)
      setPaletteCredits(null)
      return
    }
    let cancelled = false
    const fetchStatus = async () => {
      try {
        const backendUrl = await window.electronAPI.getBackendUrl()
        const [statusRes, creditsRes] = await Promise.all([
          fetch(`${backendUrl}/api/sync/status`),
          fetch(`${backendUrl}/api/sync/credits`),
        ])
        if (cancelled) return
        if (statusRes.ok) setPaletteStatus(await statusRes.json())
        if (creditsRes.ok) {
          const data = await creditsRes.json()
          setPaletteCredits(data.balance_cents ?? data.balance ?? null)
        }
      } catch { /* ignore */ }
    }
    void fetchStatus()
    const interval = setInterval(fetchStatus, 60_000)
    return () => { cancelled = true; clearInterval(interval) }
  }, [isOpen, activeTab, settings.hasPaletteApiKey])

  useEffect(() => {
    if (!isOpen || activeTab !== 'apiKeys' || !focusLtxApiKeyInputOnTabChange) return

    const frameId = window.requestAnimationFrame(() => {
      ltxApiKeyInputRef.current?.focus()
    })
    setFocusLtxApiKeyInputOnTabChange(false)

    return () => {
      window.cancelAnimationFrame(frameId)
    }
  }, [activeTab, focusLtxApiKeyInputOnTabChange, isOpen])

  // Fetch app version when About tab is shown
  useEffect(() => {
    if (activeTab !== 'about' || appVersion) return
    window.electronAPI.getAppInfo().then(info => setAppVersion(info.version)).catch(() => {})
  }, [activeTab, appVersion])

  // Fetch analytics state when modal opens
  useEffect(() => {
    if (!isOpen) return
    window.electronAPI.getAnalyticsState()
      .then((state: { analyticsEnabled: boolean }) => setAnalyticsEnabled(state.analyticsEnabled))
      .catch(() => {})
  }, [isOpen])

  // Load model data when Models tab is active
  useEffect(() => {
    if (!isOpen || activeTab !== 'models') return
    let cancelled = false
    const load = async () => {
      try {
        const backendUrl = await window.electronAPI.getBackendUrl()
        const [scanRes, guideRes] = await Promise.all([
          fetch(`${backendUrl}/api/models/video/scan`),
          fetch(`${backendUrl}/api/models/video/guide`),
        ])
        if (cancelled) return
        if (scanRes.ok) {
          const data = await scanRes.json()
          setVideoModels(data.models)
          setDistilledLoraFound(data.distilled_lora_found)
        }
        if (guideRes.ok) {
          const guide = await guideRes.json()
          setGpuInfo({ name: guide.gpu_name, vram: guide.vram_gb })
        }
      } catch (err) {
        logger.error(`Failed to load model data: ${err}`)
      }
    }
    void load()
    return () => { cancelled = true }
  }, [isOpen, activeTab])

  // Fetch text encoder status when modal opens
  useEffect(() => {
    if (!isOpen) return

    const fetchStatus = async () => {
      try {
        const backendUrl = await window.electronAPI.getBackendUrl()
        const response = await fetch(`${backendUrl}/api/models/status`)
        if (response.ok) {
          const data = await response.json()
          setTextEncoderStatus(data.text_encoder_status)
        }
      } catch (e) {
        logger.error(`Failed to fetch text encoder status: ${e}`)
      }
    }

    fetchStatus()
    // Poll while downloading
    const interval = setInterval(fetchStatus, 2000)
    return () => clearInterval(interval)
  }, [isOpen, isDownloading])

  // Handle text encoder download
  const handleDownloadTextEncoder = async () => {
    setIsDownloading(true)
    setDownloadError(null)
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const response = await fetch(`${backendUrl}/api/text-encoder/download`, { method: 'POST' })
      const data = await response.json()

      if (data.status === 'already_downloaded') {
        setTextEncoderStatus(prev => prev ? { ...prev, downloaded: true } : null)
      }
      // Poll for completion
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await fetch(`${backendUrl}/api/models/status`)
          if (statusRes.ok) {
            const statusData = await statusRes.json()
            setTextEncoderStatus(statusData.text_encoder_status)
            if (statusData.text_encoder_status?.downloaded) {
              setIsDownloading(false)
              clearInterval(pollInterval)
            }
          }
        } catch {
          // ignore
        }
      }, 2000)

      // Timeout after 30 minutes
      setTimeout(() => {
        clearInterval(pollInterval)
        if (isDownloading) setIsDownloading(false)
      }, 30 * 60 * 1000)
    } catch (e) {
      setDownloadError(e instanceof Error ? e.message : 'Download failed')
      setIsDownloading(false)
    }
  }

  if (!isOpen) return null

  const handleToggleTorchCompile = () => {
    onSettingsChange({
      ...settings,
      useTorchCompile: !settings.useTorchCompile,
    })
  }

  const handleToggleLoadOnStartup = () => {
    onSettingsChange({
      ...settings,
      loadOnStartup: !settings.loadOnStartup,
    })
  }

  const handleToggleLocalEncoder = () => {
    onSettingsChange({
      ...settings,
      useLocalTextEncoder: !settings.useLocalTextEncoder,
    })
  }

  const openApiKeysAndFocusLtxInput = () => {
    setActiveTab('apiKeys')
    setFocusLtxApiKeyInputOnTabChange(true)
  }

  const handlePromptCacheSizeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const size = Math.max(0, Math.min(1000, parseInt(e.target.value) || 100))
    onSettingsChange({
      ...settings,
      promptCacheSize: size,
    })
  }

  const handleFastUpscalerToggle = () => {
    onSettingsChange({
      ...settings,
      fastModel: { ...settings.fastModel, useUpscaler: !settings.fastModel?.useUpscaler },
    })
  }

  const handleProStepsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const steps = Math.max(1, Math.min(100, parseInt(e.target.value) || 20))
    onSettingsChange({
      ...settings,
      proModel: { ...settings.proModel, steps },
    })
  }

  const handleProUpscalerToggle = () => {
    onSettingsChange({
      ...settings,
      proModel: { ...settings.proModel, useUpscaler: !settings.proModel.useUpscaler },
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
    window.electronAPI.setAnalyticsEnabled(next).catch(() => {})
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

  const handleSyncLoras = async () => {
    setLoraSyncing(true)
    setLoraSyncResult(null)
    try {
      const backendUrl = await window.electronAPI.getBackendUrl()
      const resp = await fetch(`${backendUrl}/api/sync/library/sync-loras`, { method: 'POST' })
      const data = await resp.json()
      if (data.connected === false) {
        setLoraSyncResult('Not connected to Palette')
      } else {
        const parts: string[] = []
        if (data.synced > 0) parts.push(`${data.synced} synced`)
        if (data.skipped > 0) parts.push(`${data.skipped} already up to date`)
        if (data.failed > 0) parts.push(`${data.failed} failed`)
        setLoraSyncResult(parts.length > 0 ? parts.join(', ') : 'No LoRAs available')
      }
    } catch {
      setLoraSyncResult('Sync failed')
    } finally {
      setLoraSyncing(false)
    }
  }

  const tabs = [
    { id: 'general' as TabId, label: 'General', icon: Settings },
    { id: 'apiKeys' as TabId, label: 'API Keys', icon: KeyRound },
    { id: 'inference' as TabId, label: 'Inference', icon: Sliders },
    { id: 'promptEnhancer' as TabId, label: 'Prompt Enhancer', icon: Sparkles },
    { id: 'models' as TabId, label: 'Models', icon: Cpu },
    { id: 'about' as TabId, label: 'About', icon: Info },
  ]

  return (
    <>
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl w-full max-w-xl mx-4">
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
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
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
                        openApiKeysAndFocusLtxInput()
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
                          <span className="text-sm font-medium text-white">Generate With API</span>
                        </div>
                        <p className="text-xs text-zinc-400 mt-1">
                          Use LTX API for video generation when an LTX API key is configured.
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

              {/* Text Encoding Section */}
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
                      openApiKeysAndFocusLtxInput()
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
                      {textEncoderStatus?.downloaded ? (
                        <div className="flex items-center gap-2 text-xs text-green-400">
                          <Check className="h-4 w-4" />
                          <span>Downloaded ({textEncoderStatus.size_gb} GB)</span>
                        </div>
                      ) : isDownloading ? (
                        <div className="flex items-center gap-2 text-xs text-blue-400">
                          <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                          <span>Downloading text encoder...</span>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2 text-xs text-amber-400">
                            <AlertCircle className="h-4 w-4" />
                            <span>Not downloaded ({textEncoderStatus?.expected_size_gb || 8} GB required)</span>
                          </div>
                          <Button
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDownloadTextEncoder()
                            }}
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

              {/* Load on Startup Setting */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <svg className="h-4 w-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83" />
                      </svg>
                      <label className="text-sm font-medium text-white">
                        Preload models on startup
                      </label>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Load AI models in the background after the app starts. The video model is loaded
                      and warmed up on GPU, and the image model is preloaded into CPU RAM for faster
                      first generation. When disabled, models load on first use (faster startup, slower
                      first generation). Requires app restart to take effect.
                    </p>
                  </div>

                  {/* Toggle Switch */}
                  <button
                    onClick={handleToggleLoadOnStartup}
                    className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      settings.loadOnStartup ? 'bg-blue-500' : 'bg-zinc-700'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        settings.loadOnStartup ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>

                {/* Status indicator */}
                <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                  settings.loadOnStartup
                    ? 'bg-blue-500/10 text-blue-400'
                    : 'bg-zinc-800 text-zinc-500'
                }`}>
                  <div className={`w-1.5 h-1.5 rounded-full ${
                    settings.loadOnStartup ? 'bg-blue-400' : 'bg-zinc-600'
                  }`} />
                  {settings.loadOnStartup ? 'Models preload in background at startup' : 'Models load on first generation'}
                </div>
              </div>

              {/* Torch Compile Setting */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <svg className="h-4 w-4 text-orange-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
                      </svg>
                      <label className="text-sm font-medium text-white">
                        Torch Compile
                      </label>
                    </div>
                    <p className="text-xs text-zinc-500 leading-relaxed">
                      Compiles the model for optimized inference. <span className="text-orange-400">Experimental:</span> First
                      generation can take 5-10+ minutes for compilation. Subsequent generations may be
                      20-40% faster. Requires app restart to take effect.
                    </p>
                  </div>

                  {/* Toggle Switch */}
                  <button
                    onClick={handleToggleTorchCompile}
                    className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                      settings.useTorchCompile ? 'bg-orange-500' : 'bg-zinc-700'
                    }`}
                  >
                    <span
                      className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                        settings.useTorchCompile ? 'translate-x-5' : 'translate-x-0'
                      }`}
                    />
                  </button>
                </div>

                {/* Status indicator */}
                <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                  settings.useTorchCompile
                    ? 'bg-orange-500/10 text-orange-400'
                    : 'bg-zinc-800 text-zinc-500'
                }`}>
                  <div className={`w-1.5 h-1.5 rounded-full ${
                    settings.useTorchCompile ? 'bg-orange-400' : 'bg-zinc-600'
                  }`} />
                  {settings.useTorchCompile ? 'Optimized inference (recommended)' : 'Standard inference'}
                </div>
              </div>

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
                      Share anonymous usage data to help improve Director's Desktop.
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

          {activeTab === 'apiKeys' && (
            <>
              {/* LTX API Key Section */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Zap className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">LTX API</h3>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Your LTX API key is used for cloud text encoding, prompt enhancement, and Pro generation.
                  Add your key below to unlock these features.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex gap-2">
                    <LtxApiKeyInput
                      ref={ltxApiKeyInputRef}
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
                  <h3 className="text-sm font-semibold text-white">Replicate</h3>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">Optional</span>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Your Replicate key is used for cloud image generation when API generations are enabled.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex gap-2">
                    <LtxApiKeyInput
                      ref={replicateApiKeyInputRef}
                      value={replicateApiKeyInput}
                      onChange={(e) => setReplicateApiKeyInput(e.target.value)}
                      placeholder={settings.hasReplicateApiKey ? 'Enter new key to replace...' : 'Enter your Replicate API key...'}
                      stopPropagation
                      className="flex-1"
                    />
                    <button
                      onClick={() => {
                        const trimmed = replicateApiKeyInput.trim()
                        if (!trimmed) return
                        void saveReplicateApiKey(trimmed)
                        setReplicateApiKeyInput('')
                      }}
                      disabled={!replicateApiKeyInput.trim()}
                      className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    >
                      Save Key
                    </button>
                  </div>
                  <ApiKeyHelperRow
                    stopPropagation
                    label="Get Replicate API key"
                    onOpenKey={() => window.electronAPI.openReplicateApiKeyPage()}
                  />
                  <div className="flex items-center justify-between">
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                      settings.hasReplicateApiKey
                        ? 'bg-green-500/10 text-green-400'
                        : 'bg-zinc-800 text-zinc-500'
                    }`}>
                      {settings.hasReplicateApiKey ? (
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

                  <div className="pt-2 border-t border-zinc-700">
                    <label className="text-xs text-zinc-400 block mb-1.5">Image Model</label>
                    <select
                      value={settings.imageModel}
                      onChange={(e) => updateSettings({ imageModel: e.target.value })}
                      className="w-full bg-zinc-900 text-white text-sm rounded-lg px-3 py-2 border border-zinc-700 focus:border-blue-500 focus:outline-none"
                    >
                      <option value="flux-klein-9b">FLUX.2 Klein 9B</option>
                      <option value="z-image-turbo">Z-Image Turbo</option>
                      <option value="nano-banana-2">Nano Banana 2 (Cloud)</option>
                    </select>
                    <p className="text-[11px] text-zinc-500 mt-1.5 leading-relaxed">
                      {settings.imageModel === 'flux-klein-9b' && 'High quality images with LoRA support. Reloads each generation (~5s extra).'}
                      {settings.imageModel === 'z-image-turbo' && 'Fast single-step generation. Stays in memory between runs for quick back-to-back images.'}
                      {settings.imageModel === 'nano-banana-2' && 'Runs in the cloud — no GPU needed. Requires a Replicate API key.'}
                      {' '}Switching models takes a few seconds on your first image — you\'ll see a progress message.
                    </p>
                  </div>

                  <div className="pt-2 border-t border-zinc-700">
                    <label className="text-xs text-zinc-400 block mb-1.5">Video Model</label>
                    <select
                      value={settings.videoModel}
                      onChange={(e) => updateSettings({ videoModel: e.target.value })}
                      className="w-full bg-zinc-900 text-white text-sm rounded-lg px-3 py-2 border border-zinc-700 focus:border-blue-500 focus:outline-none"
                    >
                      <option value="ltx-fast">LTX Fast</option>
                      <option value="seedance-1.5-pro">Seedance 1.5 Pro</option>
                    </select>
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
                  Your Gemini API key is used for AI-powered prompt suggestions when filling timeline gaps.
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

              {/* CivitAI API Key Section */}
              <div className="space-y-4 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <Download className="h-4 w-4 text-orange-400" />
                  <h3 className="text-sm font-semibold text-white">CivitAI</h3>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">LoRA Browser</span>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Add your CivitAI API key to browse and download LoRAs directly from the app.
                </p>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                  <div className="flex gap-2">
                    <input
                      type="password"
                      value={civitaiApiKeyInput}
                      onChange={(e) => setCivitaiApiKeyInput(e.target.value)}
                      placeholder={settings.hasCivitaiApiKey ? 'Enter new key to replace...' : 'Enter your CivitAI API key...'}
                      onKeyDown={(e) => e.stopPropagation()}
                      className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                    />
                    <button
                      onClick={() => {
                        const trimmed = civitaiApiKeyInput.trim()
                        if (!trimmed) return
                        void saveCivitaiApiKey(trimmed)
                        setCivitaiApiKeyInput('')
                      }}
                      disabled={!civitaiApiKeyInput.trim()}
                      className="px-3 py-2 bg-orange-600 text-white text-sm rounded-lg hover:bg-orange-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                    >
                      Save Key
                    </button>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
                      settings.hasCivitaiApiKey
                        ? 'bg-green-500/10 text-green-400'
                        : 'bg-zinc-800 text-zinc-500'
                    }`}>
                      {settings.hasCivitaiApiKey ? (
                        <>
                          <Check className="h-3 w-3" />
                          Key configured
                        </>
                      ) : (
                        <>
                          <AlertCircle className="h-3 w-3" />
                          Optional — needed for CivitAI browsing
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Director's Palette Section */}
              <div className="space-y-4 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <Film className="h-4 w-4 text-amber-400" />
                  <h3 className="text-sm font-semibold text-white">Director's Palette</h3>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">Cloud Sync</span>
                </div>

                <p className="text-xs text-zinc-500 leading-relaxed">
                  Connect to Director's Palette to sync your gallery, characters, and library between web and desktop.
                </p>

                {paletteStatus?.connected ? (
                  <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 bg-green-500/10 text-green-400">
                        <Check className="h-3 w-3" />
                        Connected{paletteStatus.user ? ` as ${paletteStatus.user.email}` : ''}
                      </div>
                      <button
                        onClick={async () => {
                          try {
                            const backendUrl = await window.electronAPI.getBackendUrl()
                            await fetch(`${backendUrl}/api/sync/disconnect`, { method: 'POST' })
                            setPaletteStatus(null)
                            setPaletteCredits(null)
                            void refreshSettings()
                          } catch { /* ignore */ }
                        }}
                        className="text-xs text-zinc-500 hover:text-red-400 transition-colors"
                      >
                        Disconnect
                      </button>
                    </div>
                    {paletteCredits !== null && (
                      <span className="text-xs text-zinc-400">
                        Credits: <span className="text-white font-medium">{paletteCredits.toLocaleString()}</span>
                      </span>
                    )}
                    <div className="flex items-center gap-3 mt-2">
                      <button
                        onClick={handleSyncLoras}
                        disabled={loraSyncing}
                        className="px-3 py-1.5 rounded-lg text-xs bg-purple-600 hover:bg-purple-500 text-white disabled:opacity-50 transition-colors"
                      >
                        {loraSyncing ? 'Syncing LoRAs...' : 'Sync LoRAs'}
                      </button>
                      {loraSyncResult && (
                        <span className="text-xs text-zinc-400">{loraSyncResult}</span>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                    {/* Auth mode tabs */}
                    <div className="flex items-center bg-zinc-900 rounded-lg border border-zinc-800 p-0.5 w-fit">
                      <button
                        onClick={() => { setPaletteAuthMode('login'); setPaletteLoginError(null) }}
                        className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                          paletteAuthMode === 'login' ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-white'
                        }`}
                      >
                        Login with Email
                      </button>
                      <button
                        onClick={() => { setPaletteAuthMode('apikey'); setPaletteLoginError(null) }}
                        className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                          paletteAuthMode === 'apikey' ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-white'
                        }`}
                      >
                        API Key
                      </button>
                    </div>

                    {paletteAuthMode === 'login' ? (
                      <div className="space-y-2">
                        <input
                          type="email"
                          value={paletteLoginEmail}
                          onChange={(e) => setPaletteLoginEmail(e.target.value)}
                          placeholder="Email address"
                          onKeyDown={(e) => { e.stopPropagation(); if (e.key === 'Enter') document.getElementById('palette-password')?.focus() }}
                          className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                        />
                        <input
                          id="palette-password"
                          type="password"
                          value={paletteLoginPassword}
                          onChange={(e) => setPaletteLoginPassword(e.target.value)}
                          placeholder="Password"
                          onKeyDown={(e) => {
                            e.stopPropagation()
                            if (e.key === 'Enter' && paletteLoginEmail.trim() && paletteLoginPassword) {
                              (e.target as HTMLInputElement).closest('div')?.querySelector('button')?.click()
                            }
                          }}
                          className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                        />
                        <button
                          onClick={async () => {
                            const email = paletteLoginEmail.trim()
                            const password = paletteLoginPassword
                            if (!email || !password) return
                            setPaletteLoginLoading(true)
                            setPaletteLoginError(null)
                            try {
                              const backendUrl = await window.electronAPI.getBackendUrl()
                              const res = await fetch(`${backendUrl}/api/sync/login`, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ email, password }),
                              })
                              const data = await res.json()
                              if (data.connected) {
                                setPaletteStatus(data)
                                setPaletteLoginEmail('')
                                setPaletteLoginPassword('')
                                void refreshSettings()
                              } else {
                                setPaletteLoginError(data.error || 'Login failed')
                              }
                            } catch (err) {
                              setPaletteLoginError(err instanceof Error ? err.message : 'Login failed')
                            } finally {
                              setPaletteLoginLoading(false)
                            }
                          }}
                          disabled={!paletteLoginEmail.trim() || !paletteLoginPassword || paletteLoginLoading}
                          className="w-full px-3 py-2 bg-amber-600 text-white text-sm font-medium rounded-lg hover:bg-amber-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors"
                        >
                          {paletteLoginLoading ? 'Signing in...' : 'Sign In'}
                        </button>
                      </div>
                    ) : (
                      <div className="flex gap-2">
                        <input
                          type="password"
                          value={paletteApiKeyInput}
                          onChange={(e) => setPaletteApiKeyInput(e.target.value)}
                          placeholder="Enter your Palette API key (dp_...)..."
                          onKeyDown={(e) => e.stopPropagation()}
                          className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                        />
                        <button
                          onClick={async () => {
                            const trimmed = paletteApiKeyInput.trim()
                            if (!trimmed) return
                            setPaletteLoginLoading(true)
                            setPaletteLoginError(null)
                            try {
                              const backendUrl = await window.electronAPI.getBackendUrl()
                              const res = await fetch(`${backendUrl}/api/sync/connect`, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ token: trimmed }),
                              })
                              const data = await res.json()
                              if (data.connected) {
                                setPaletteStatus(data)
                                setPaletteApiKeyInput('')
                                void refreshSettings()
                              } else {
                                setPaletteLoginError(data.error || 'Connection failed')
                              }
                            } catch (err) {
                              setPaletteLoginError(err instanceof Error ? err.message : 'Connection failed')
                            } finally {
                              setPaletteLoginLoading(false)
                            }
                          }}
                          disabled={!paletteApiKeyInput.trim() || paletteLoginLoading}
                          className="px-3 py-2 bg-amber-600 text-white text-sm rounded-lg hover:bg-amber-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                        >
                          {paletteLoginLoading ? 'Connecting...' : 'Connect'}
                        </button>
                      </div>
                    )}

                    {/* Status / Error */}
                    {paletteLoginError && (
                      <div className="text-xs px-2 py-1.5 rounded bg-red-500/10 text-red-400 flex items-start gap-1.5">
                        <AlertCircle className="h-3 w-3 mt-0.5 shrink-0" />
                        <span>{paletteLoginError}</span>
                      </div>
                    )}

                    {!paletteLoginError && paletteStatus && !paletteStatus.connected && paletteStatus.error && (
                      <div className="text-xs px-2 py-1.5 rounded bg-amber-500/10 text-amber-400 flex items-start gap-1.5">
                        <AlertCircle className="h-3 w-3 mt-0.5 shrink-0" />
                        <span>{paletteStatus.error}</span>
                      </div>
                    )}

                    {!paletteLoginError && !paletteStatus?.connected && !settings.hasPaletteApiKey && (
                      <div className="text-xs px-2 py-1 rounded bg-zinc-800 text-zinc-500 inline-flex items-center gap-1.5">
                        <AlertCircle className="h-3 w-3" />
                        Optional
                      </div>
                    )}
                  </div>
                )}
              </div>
            </>
          )}

          {activeTab === 'inference' && (
            <>
              {/* Fast Model Settings */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Zap className="h-4 w-4 text-green-400" />
                  <h3 className="text-sm font-semibold text-white">Fast Model (Distilled)</h3>
                </div>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-4">
                  {/* Steps Info */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">Inference Steps</label>
                      <p className="text-xs text-zinc-500">Fixed at 8 steps (built into distilled model)</p>
                    </div>
                    <span className="px-3 py-1.5 bg-zinc-700 rounded-lg text-sm text-zinc-400">8</span>
                  </div>

                  {/* Upscaler Toggle */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">2x Upscaler</label>
                      <p className="text-xs text-zinc-500">When off, generates at native resolution</p>
                    </div>
                    <button
                      onClick={handleFastUpscalerToggle}
                      className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                        settings.fastModel?.useUpscaler !== false ? 'bg-green-500' : 'bg-zinc-700'
                      }`}
                    >
                      <span
                        className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                          settings.fastModel?.useUpscaler !== false ? 'translate-x-5' : 'translate-x-0'
                        }`}
                      />
                    </button>
                  </div>
                </div>

                {/* Summary */}
                <div className="text-xs text-zinc-500">
                  Current: 8 steps, {settings.fastModel?.useUpscaler !== false ? 'with upscaler (2-stage, recommended)' : 'native resolution (experimental)'}
                </div>
              </div>

              {/* Pro Model Settings */}
              <div className="space-y-4 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <svg className="h-4 w-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                  </svg>
                  <h3 className="text-sm font-semibold text-white">Pro Model (Full)</h3>
                </div>

                <div className="bg-zinc-800/50 rounded-lg p-4 space-y-4">
                  {/* Steps */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">Inference Steps</label>
                      <p className="text-xs text-zinc-500">More steps = better quality, slower</p>
                    </div>
                    <input
                      type="number"
                      min="1"
                      max="100"
                      value={settings.proModel?.steps ?? 20}
                      onChange={handleProStepsChange}
                      className="w-20 px-3 py-1.5 bg-zinc-700 border border-zinc-600 rounded-lg text-sm text-white text-center focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>

                  {/* Upscaler Toggle */}
                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm text-white">2x Upscaler</label>
                      <p className="text-xs text-zinc-500">Doubles resolution in second pass</p>
                    </div>
                    <button
                      onClick={handleProUpscalerToggle}
                      className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
                        settings.proModel?.useUpscaler !== false ? 'bg-blue-500' : 'bg-zinc-700'
                      }`}
                    >
                      <span
                        className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                          settings.proModel?.useUpscaler !== false ? 'translate-x-5' : 'translate-x-0'
                        }`}
                      />
                    </button>
                  </div>
                </div>

                {/* Summary */}
                <div className="text-xs text-zinc-500">
                  Current: {settings.proModel?.steps ?? 20} steps, {settings.proModel?.useUpscaler !== false ? 'with upscaler (2-stage, recommended)' : 'native resolution'}
                </div>
              </div>

              {/* Info Box */}
              <div className="bg-zinc-800/30 rounded-lg p-3 mt-4">
                <p className="text-xs text-zinc-400">
                  <span className="text-blue-400 font-medium">Tip:</span> Lower steps = faster but lower quality.
                  Higher steps = better quality but slower.
                </p>
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

          {activeTab === 'models' && (
            <>
              {/* GPU Info Banner */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Cpu className="h-4 w-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">GPU Info</h3>
                </div>
                <div className="bg-zinc-800/50 rounded-lg p-4 border border-zinc-700/50">
                  {gpuInfo ? (
                    <div className="flex items-center gap-3">
                      <Cpu className="h-5 w-5 text-blue-400 flex-shrink-0" />
                      <div>
                        <p className="text-sm font-medium text-white">{gpuInfo.name ?? 'Unknown GPU'}</p>
                        {gpuInfo.vram !== null && (
                          <p className="text-xs text-zinc-400">{gpuInfo.vram} GB VRAM</p>
                        )}
                      </div>
                    </div>
                  ) : (
                    <p className="text-sm text-zinc-500">Loading GPU info...</p>
                  )}
                </div>
              </div>

              {/* Video Model Selection */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <Film className="h-4 w-4 text-purple-400" />
                  <h3 className="text-sm font-semibold text-white">Video Model</h3>
                </div>
                <select
                  value={settings.selectedVideoModel}
                  onChange={async (e) => {
                    const model = e.target.value
                    updateSettings({ selectedVideoModel: model })
                    try {
                      const backendUrl = await window.electronAPI.getBackendUrl()
                      await fetch(`${backendUrl}/api/models/video/select`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ model }),
                      })
                    } catch (err) {
                      logger.error(`Failed to select video model: ${err}`)
                    }
                  }}
                  className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">-- Select a model --</option>
                  {videoModels.map((m: any) => (
                    <option key={m.path ?? m.filename ?? m.name} value={m.path ?? m.filename ?? m.name}>
                      {m.display_name ?? m.filename ?? m.name}
                    </option>
                  ))}
                </select>
                {videoModels.length === 0 && (
                  <p className="text-xs text-zinc-500">No model files detected. Scan your model folder to find models.</p>
                )}
              </div>

              {/* Model Folder */}
              <div className="space-y-3 pt-4 border-t border-zinc-800">
                <div className="flex items-center gap-2">
                  <FolderOpen className="h-4 w-4 text-amber-400" />
                  <h3 className="text-sm font-semibold text-white">Model Folder</h3>
                </div>
                <div className="bg-zinc-800/50 rounded-lg p-3 border border-zinc-700/50">
                  <p className="text-xs text-zinc-400 font-mono break-all">
                    {settings.customVideoModelPath || 'Default model folder'}
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={async () => {
                      const result = await window.electronAPI?.showOpenDirectoryDialog?.({ title: 'Select Video Models Folder' })
                      if (result) {
                        updateSettings({ customVideoModelPath: result })
                      }
                    }}
                    className="flex-1 px-3 py-2 bg-zinc-700 hover:bg-zinc-600 text-white text-xs rounded-lg transition-colors"
                  >
                    Change
                  </button>
                  <button
                    onClick={() => {
                      const folderPath = settings.customVideoModelPath
                      if (folderPath) {
                        window.electronAPI?.showItemInFolder(folderPath)
                      }
                    }}
                    disabled={!settings.customVideoModelPath}
                    className="flex-1 px-3 py-2 bg-zinc-700 hover:bg-zinc-600 text-white text-xs rounded-lg transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    Open Folder
                  </button>
                  <button
                    onClick={async () => {
                      setModelScanning(true)
                      try {
                        const backendUrl = await window.electronAPI.getBackendUrl()
                        const scanRes = await fetch(`${backendUrl}/api/models/video/scan`)
                        if (scanRes.ok) {
                          const data = await scanRes.json()
                          setVideoModels(data.models)
                          setDistilledLoraFound(data.distilled_lora_found)
                        }
                      } catch (err) {
                        logger.error(`Failed to scan models: ${err}`)
                      } finally {
                        setModelScanning(false)
                      }
                    }}
                    disabled={modelScanning}
                    className="flex-1 px-3 py-2 bg-zinc-700 hover:bg-zinc-600 text-white text-xs rounded-lg transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-1.5"
                  >
                    <RefreshCw className={`h-3 w-3 ${modelScanning ? 'animate-spin' : ''}`} />
                    {modelScanning ? 'Scanning...' : 'Scan'}
                  </button>
                </div>
              </div>

              {/* Distilled LoRA Warning */}
              {settings.selectedVideoModel && !distilledLoraFound && (
                <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4">
                  <div className="flex items-start gap-2.5">
                    <AlertCircle className="h-4 w-4 text-amber-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="text-sm text-amber-300 font-medium">Speed Boost file missing</p>
                      <p className="text-xs text-zinc-400 mt-1 leading-relaxed">
                        Compressed models need a small extra file (the "Speed Boost LoRA") to generate videos quickly.
                        Open the Model Guide below to find the download link — just drop it in your models folder.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Model Guide Button */}
              <div className="pt-2">
                <button
                  onClick={() => setShowModelGuide(true)}
                  className="w-full px-4 py-2.5 bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/30 text-blue-400 text-sm font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  <Info className="h-4 w-4" />
                  Which Model Do I Need?
                </button>
              </div>
            </>
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
                    <h3 className="text-lg font-bold text-white">Director's Desktop</h3>
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
    <ModelGuideDialog isOpen={showModelGuide} onClose={() => setShowModelGuide(false)} />
    </>
  )
}

export type { AppSettings, TabId as SettingsTabId }
