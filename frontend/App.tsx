import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Loader2, AlertCircle, Settings, FileText, Monitor, Server } from 'lucide-react'
import { ApiClient, type ApiSuccessOf } from './lib/api-client'
import { ProjectProvider } from './contexts/ProjectContext'
import { ViewProvider, useView } from './contexts/ViewContext'
import { KeyboardShortcutsProvider } from './contexts/KeyboardShortcutsContext'
import { AppSettingsProvider, useAppSettings } from './contexts/AppSettingsContext'
import { KeyboardShortcutsModal } from './components/KeyboardShortcutsModal'
import { useBackend } from './hooks/use-backend'
import { logger } from './lib/logger'
import { Home } from './views/Home'
import { Project } from './views/Project'
import { LaunchGate } from './components/FirstRunSetup'
import { LtxUpgradePrompt } from './components/LtxUpgradePrompt'
import { PythonSetup } from './components/PythonSetup'
import { SettingsModal, type SettingsTabId } from './components/SettingsModal'
import { LogViewer } from './components/LogViewer'
import { ApiGatewayModal, type ApiGatewaySection } from './components/ApiGatewayModal'
import { Button } from './components/ui/button'
import { BackendConnectionPanel } from './components/BackendConnectionPanel'

type SetupState = 'loading' | { needsSetup: boolean; needsLicense: boolean }
type RequiredModelsGateState = 'checking' | 'missing' | 'ready'
type ConfiguredBackendMode = 'managed-local' | 'external'
type LtxRecommendation = ApiSuccessOf<'getLtxRecommendation'>
type LtxUpgradeRecommendation = Extract<LtxRecommendation, { status: 'upgrade' }>

function AppContent() {
  const { currentView } = useView()
  const {
    connected,
    processStatus,
    isLoading: backendLoading,
    connectionMode,
    message: backendMessage,
  } = useBackend()
  const { settings, saveLtxApiKey, saveFalApiKey, forceApiGenerations, isLoaded, runtimePolicyLoaded } = useAppSettings()

  const [pythonReady, setPythonReady] = useState<boolean | null>(null)
  const [configuredBackendMode, setConfiguredBackendMode] = useState<ConfiguredBackendMode | null>(null)
  const [configuredBackendUrl, setConfiguredBackendUrl] = useState('')
  const [pythonSetupSelected, setPythonSetupSelected] = useState(false)
  const [firstRunComputeConfirmed, setFirstRunComputeConfirmed] = useState(false)
  const [backendStarted, setBackendStarted] = useState(false)
  const [setupState, setSetupState] = useState<SetupState>('loading')
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [settingsInitialTab, setSettingsInitialTab] = useState<SettingsTabId | undefined>(undefined)
  const [isLogViewerOpen, setIsLogViewerOpen] = useState(false)
  const [isFinalizingFirstRun, setIsFinalizingFirstRun] = useState(false)
  const [firstRunFinalizeError, setFirstRunFinalizeError] = useState<string | null>(null)
  const [requiredModelsGate, setRequiredModelsGate] = useState<RequiredModelsGateState>('checking')
  const [ltxUpgradeRecommendation, setLtxUpgradeRecommendation] = useState<LtxUpgradeRecommendation | null>(null)
  const [dismissedUpgradeTargetId, setDismissedUpgradeTargetId] = useState<LtxUpgradeRecommendation['ltx_model_id'] | null>(
    null,
  )
  const setupCompletionInFlightRef = useRef<Promise<void> | null>(null)

  type ApiGatewayRequest = {
    requiredKeys: Array<'ltx' | 'fal'>
    title: string
    description: string
    blocking?: boolean
    includeOptionalMissing?: boolean
  }

  const [apiGatewayRequest, setApiGatewayRequest] = useState<ApiGatewayRequest | null>(null)

  const isBackendRestarting = processStatus === 'restarting' && connectionMode !== 'external'
  const isBackendDead = processStatus === 'dead' && connectionMode !== 'external'
  const isExternalBackendUnreachable = processStatus === 'unreachable' && connectionMode === 'external'
  const waitingForRuntimePolicy = processStatus === 'alive' && !runtimePolicyLoaded

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail
      if (detail?.tab) setSettingsInitialTab(detail.tab)
      setIsSettingsOpen(true)
    }
    window.addEventListener('open-settings', handler)
    return () => window.removeEventListener('open-settings', handler)
  }, [])

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail ?? {}
      const requiredKeys = Array.isArray(detail.requiredKeys) ? detail.requiredKeys : ['ltx']
      setApiGatewayRequest({
        requiredKeys,
        title: detail.title ?? 'Connect API Keys',
        description: detail.description ?? 'Add the required API keys to continue.',
        blocking: detail.blocking ?? false,
        includeOptionalMissing: detail.includeOptionalMissing ?? false,
      })
    }
    window.addEventListener('open-api-gateway', handler)
    return () => window.removeEventListener('open-api-gateway', handler)
  }, [])

  useEffect(() => {
    const check = async () => {
      try {
        const connection = await window.electronAPI.getBackendConnectionConfig()
        setConfiguredBackendMode(connection.mode)
        setConfiguredBackendUrl(connection.url)
        const result = await window.electronAPI.checkPythonReady()
        setPythonReady(result.ready)
      } catch (e) {
        logger.error(`Failed to check Python readiness: ${e}`)
        setPythonReady(false)
      }
    }
    void check()
  }, [])

  useEffect(() => {
    if (pythonReady !== true || backendStarted) return
    setBackendStarted(true)
    const start = async () => {
      try {
        logger.info('Starting backend connection...')
        await window.electronAPI.startBackendConnection()
        logger.info('Backend connection started successfully')
      } catch (e) {
        logger.error(`Failed to start Python backend: ${e}`)
      }
    }
    void start()
  }, [pythonReady, backendStarted])

  useEffect(() => {
    const checkFirstRun = async () => {
      try {
        const next = await window.electronAPI.checkFirstRun()
        setSetupState(next)
      } catch (e) {
        logger.error(`Failed to check first run: ${e}`)
        setSetupState({ needsSetup: false, needsLicense: false })
      }
    }
    void checkFirstRun()
  }, [])

  const handleFirstRunComplete = useCallback(async () => {
    if (setupCompletionInFlightRef.current) {
      return setupCompletionInFlightRef.current
    }

    setFirstRunFinalizeError(null)
    setIsFinalizingFirstRun(true)

    const inFlightPromise = (async () => {
      const ok = await window.electronAPI.completeSetup()
      if (!ok) {
        throw new Error('Failed to complete setup.')
      }
      setSetupState({ needsSetup: false, needsLicense: false })
    })()

    setupCompletionInFlightRef.current = inFlightPromise

    try {
      await inFlightPromise
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Failed to finalize setup.'
      setFirstRunFinalizeError(message)
      throw e
    } finally {
      setupCompletionInFlightRef.current = null
      setIsFinalizingFirstRun(false)
    }
  }, [])

  const handleAcceptLicense = useCallback(async () => {
    const ok = await window.electronAPI.acceptLicense()
    if (!ok) {
      throw new Error('Failed to save license acceptance.')
    }
    setSetupState((prev) => {
      if (prev === 'loading') return prev
      return { ...prev, needsLicense: false }
    })
  }, [])

  const saveApiKeyForFirstRun = useCallback(
    async (apiKey: string) => {
      const trimmed = apiKey.trim()
      if (!trimmed) {
        throw new Error('Please enter a valid LTX API key.')
      }

      await saveLtxApiKey(trimmed)
      setFirstRunFinalizeError(null)
    },
    [saveLtxApiKey],
  )

  const isForcedFirstRun =
    setupState !== 'loading' && setupState.needsSetup && !setupState.needsLicense && forceApiGenerations

  const shouldAutoFinalizeForcedFirstRun =
    isForcedFirstRun && isLoaded && settings.hasLtxApiKey && !isFinalizingFirstRun && !firstRunFinalizeError

  const areRequiredModelsDownloaded = useCallback(async () => {
    const [ltxResult, imgGenResult] = await Promise.all([
      ApiClient.getLtxRecommendation(),
      ApiClient.getImgGenRecommendation(),
    ])
    if (!ltxResult.ok) {
      throw new Error(ltxResult.error.message)
    }
    if (!imgGenResult.ok) {
      throw new Error(imgGenResult.error.message)
    }
    return ltxResult.data.status !== 'download' && imgGenResult.data.cp_to_download === null
  }, [])

  const handleMissingModelsComplete = useCallback(async () => {
    const allDownloaded = await areRequiredModelsDownloaded()
    if (!allDownloaded) {
      throw new Error('Required models are still missing. Please finish downloading before continuing.')
    }
    await handleFirstRunComplete()
    setRequiredModelsGate('ready')
  }, [areRequiredModelsDownloaded, handleFirstRunComplete])

  useEffect(() => {
    if (!shouldAutoFinalizeForcedFirstRun) return
    void handleFirstRunComplete().catch(() => {
      // Error state is handled via firstRunFinalizeError.
    })
  }, [shouldAutoFinalizeForcedFirstRun, handleFirstRunComplete])

  useEffect(() => {
    if (setupState === 'loading' || waitingForRuntimePolicy || backendLoading || !connected) {
      return
    }

    if (forceApiGenerations || setupState.needsLicense || setupState.needsSetup) {
      setRequiredModelsGate('ready')
      return
    }

    let cancelled = false
    setRequiredModelsGate('checking')

    const checkRequiredModels = async () => {
      try {
        const allDownloaded = await areRequiredModelsDownloaded()
        if (cancelled) return
        setRequiredModelsGate(allDownloaded ? 'ready' : 'missing')
      } catch (e) {
        logger.error(`Failed to check required model status: ${e}`)
        if (cancelled) return
        // Do not block app launch on transient status-check failures.
        setRequiredModelsGate('ready')
      }
    }

    void checkRequiredModels()

    return () => {
      cancelled = true
    }
  }, [
    areRequiredModelsDownloaded,
    backendLoading,
    forceApiGenerations,
    setupState,
    connected,
    waitingForRuntimePolicy,
  ])

  const refreshLtxUpgradeRecommendation = useCallback(async () => {
    const result = await ApiClient.getLtxRecommendation()
    if (!result.ok) {
      logger.warn(`Failed to fetch LTX upgrade recommendation: ${result.error.message}`)
      setLtxUpgradeRecommendation(null)
      return
    }

    const recommendation = result.data
    if (recommendation.status === 'upgrade' && recommendation.ltx_model_id !== dismissedUpgradeTargetId) {
      setLtxUpgradeRecommendation(recommendation)
      return
    }
    setLtxUpgradeRecommendation(null)
  }, [dismissedUpgradeTargetId])

  useEffect(() => {
    if (
      backendLoading
      || setupState === 'loading'
      || waitingForRuntimePolicy
      || !connected
      || forceApiGenerations
      || setupState.needsLicense
      || setupState.needsSetup
      || requiredModelsGate !== 'ready'
    ) {
      setLtxUpgradeRecommendation(null)
      return
    }

    let cancelled = false
    const loadRecommendation = async () => {
      const result = await ApiClient.getLtxRecommendation()
      if (cancelled) return
      if (!result.ok) {
        logger.warn(`Failed to fetch LTX upgrade recommendation: ${result.error.message}`)
        setLtxUpgradeRecommendation(null)
        return
      }

      const recommendation = result.data
      if (recommendation.status === 'upgrade' && recommendation.ltx_model_id !== dismissedUpgradeTargetId) {
        setLtxUpgradeRecommendation(recommendation)
        return
      }

      setLtxUpgradeRecommendation(null)
    }

    void loadRecommendation()

    return () => {
      cancelled = true
    }
  }, [
    backendLoading,
    dismissedUpgradeTargetId,
    forceApiGenerations,
    requiredModelsGate,
    setupState,
    connected,
    waitingForRuntimePolicy,
  ])

  const handleDismissLtxUpgradePrompt = useCallback(() => {
    if (!ltxUpgradeRecommendation) return
    setDismissedUpgradeTargetId(ltxUpgradeRecommendation.ltx_model_id)
    setLtxUpgradeRecommendation(null)
  }, [ltxUpgradeRecommendation])

  const handleCompleteLtxUpgradePrompt = useCallback(async () => {
    setDismissedUpgradeTargetId(null)
    await refreshLtxUpgradeRecommendation()
  }, [refreshLtxUpgradeRecommendation])

  const restartingOverlay = isBackendRestarting ? (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="rounded-lg border border-zinc-700 bg-zinc-900/95 px-6 py-4 text-center shadow-xl">
        <div className="flex items-center justify-center gap-2 text-zinc-100">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="font-medium">Reconnecting...</span>
        </div>
        <p className="mt-2 text-sm text-zinc-400">The backend process stopped unexpectedly. Attempting to restart...</p>
      </div>
    </div>
  ) : null

  const externalBackendBanner = isExternalBackendUnreachable ? (
    <div className="fixed left-1/2 top-3 z-[70] flex w-[min(44rem,calc(100%-1.5rem))] -translate-x-1/2 items-center gap-3 rounded-lg border border-amber-500/30 bg-zinc-900/95 px-4 py-3 shadow-2xl backdrop-blur-sm">
      <AlertCircle className="h-5 w-5 flex-shrink-0 text-amber-400" />
      <div className="min-w-0 flex-1">
        <div className="text-sm font-medium text-zinc-100">Remote backend disconnected</div>
        <p className="truncate text-xs text-zinc-400" title={backendMessage ?? undefined}>
          {backendMessage || 'Local editing remains available while LTX Desktop reconnects.'}
        </p>
      </div>
      <Button
        variant="outline"
        className="h-8 flex-shrink-0 border-zinc-700 px-3 text-xs"
        onClick={() => { void window.electronAPI.retryBackendConnection() }}
      >
        Retry
      </Button>
      <Button
        variant="outline"
        className="h-8 flex-shrink-0 border-zinc-700 px-3 text-xs"
        onClick={() => {
          setSettingsInitialTab('compute')
          setIsSettingsOpen(true)
        }}
      >
        Compute settings
      </Button>
    </div>
  ) : null

  const showGlobalControls = connected && setupState !== 'loading' && !setupState.needsSetup
  const shouldBlockUntilSettingsLoaded = forceApiGenerations && !isLoaded
  const shouldShowForcedFirstRunUpsell = isForcedFirstRun && isLoaded && !settings.hasLtxApiKey
  const shouldShowGlobalForcedUpsell = forceApiGenerations && setupState !== 'loading' && !setupState.needsSetup && isLoaded && !settings.hasLtxApiKey
  const shouldBlockForLtxKey = shouldShowForcedFirstRunUpsell || shouldShowGlobalForcedUpsell
  const forcedApiGatewayRequest: ApiGatewayRequest | null = shouldBlockForLtxKey
    ? {
        requiredKeys: ['ltx'],
        title: 'Connect API Keys',
        description: 'This app is configured for API-only generation. Add your API key to continue.',
        blocking: true,
        includeOptionalMissing: true,
      }
    : null
  const activeApiGatewayRequest = apiGatewayRequest ?? forcedApiGatewayRequest
  const shouldShowGateway = activeApiGatewayRequest !== null

  const gatewaySections: ApiGatewaySection[] = useMemo(() => {
    if (!activeApiGatewayRequest) return []

    const handleSaveLtxKey = async (apiKey: string) => {
      if (isForcedFirstRun) {
        await saveApiKeyForFirstRun(apiKey)
        return
      }
      await saveLtxApiKey(apiKey)
    }

    const sections: ApiGatewaySection[] = [
      {
        keyType: 'ltx',
        title: 'LTX Cloud API',
        description: 'Video generation, prompt enhancement, and cloud text encoding.',
        required: activeApiGatewayRequest.requiredKeys.includes('ltx'),
        isConfigured: settings.hasLtxApiKey,
        inputLabel: 'LTX Cloud API key',
        placeholder: 'Enter your LTX Cloud API key...',
        onSave: handleSaveLtxKey,
        onGetKey: () => window.electronAPI.openLtxApiKeyPage(),
        getKeyLabel: 'Get LTX Cloud API key',
      },
      {
        keyType: 'fal',
        title: 'FAL AI',
        description: 'Required to generate images with Z Image Turbo.',
        required: activeApiGatewayRequest.requiredKeys.includes('fal'),
        isConfigured: settings.hasFalApiKey,
        inputLabel: 'FAL AI API key',
        placeholder: 'Enter your FAL AI API key...',
        onSave: saveFalApiKey,
        onGetKey: () => window.electronAPI.openFalApiKeyPage(),
        getKeyLabel: 'Get FAL API key',
      },
    ]

    return sections.filter((section) => {
      if (section.required) return true
      if (activeApiGatewayRequest.includeOptionalMissing) return true
      return false
    })
  }, [
    activeApiGatewayRequest,
    isForcedFirstRun,
    saveApiKeyForFirstRun,
    saveFalApiKey,
    saveLtxApiKey,
    settings.hasFalApiKey,
    settings.hasLtxApiKey,
  ])

  if (pythonReady === null) {
    return (
      <div className="h-screen bg-background flex items-center justify-center">
        <Loader2 className="h-8 w-8 text-primary animate-spin" />
      </div>
    )
  }

  if (pythonReady === false) {
    if (configuredBackendMode === 'external' || pythonSetupSelected) {
      return <PythonSetup onReady={() => setPythonReady(true)} />
    }
    return (
      <div className="h-screen overflow-auto bg-background p-6">
        <div className="mx-auto flex min-h-full w-full max-w-2xl items-center justify-center">
          <div className="w-full space-y-5">
            <div className="rounded-xl border border-zinc-700 bg-zinc-900/70 p-5 text-center">
              <h2 className="text-xl font-semibold text-white">Choose where inference runs</h2>
              <p className="mx-auto mt-2 max-w-lg text-sm text-zinc-400">
                LTX Desktop needs local media tools for project assets and export, even when inference runs on another machine.
              </p>
              <Button className="mt-4" onClick={() => setPythonSetupSelected(true)}>
                Install local media tools
              </Button>
            </div>
            <BackendConnectionPanel />
          </div>
        </div>
      </div>
    )
  }

  if (isBackendDead) {
    return (
      <div className="h-screen bg-background flex items-center justify-center p-6">
        <div className="w-full max-w-5xl rounded-xl border border-zinc-700 bg-zinc-900/80 p-6 shadow-2xl">
          <div className="text-center">
            <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-foreground mb-2">The backend process crashed and could not be restarted</h2>
            <p className="text-muted-foreground mb-4">Review the logs below and restart the application.</p>
          </div>
          <div className="h-[50vh]">
            <LogViewer isOpen={true} onClose={() => {}} embedded={true} />
          </div>
          <div className="mt-4 flex justify-center">
            <Button onClick={() => window.location.reload()}>Restart Application</Button>
          </div>
        </div>
      </div>
    )
  }

  const waitingForRequiredModels =
    requiredModelsGate === 'checking' &&
    connected &&
    setupState !== 'loading' &&
    !waitingForRuntimePolicy &&
    !forceApiGenerations

  if (backendLoading || setupState === 'loading' || waitingForRuntimePolicy || waitingForRequiredModels) {
    return (
      <div className="relative h-screen w-screen">
        <div className="h-screen bg-background flex items-center justify-center">
          <div className="text-center">
            <Loader2 className="h-12 w-12 text-primary animate-spin mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-foreground mb-2">Starting LTX Desktop...</h2>
            <p className="text-muted-foreground">Initializing the inference engine</p>
          </div>
        </div>
        {restartingOverlay}
        {externalBackendBanner}
      </div>
    )
  }

  if (setupState.needsLicense) {
    const licenseOnly = forceApiGenerations || !setupState.needsSetup
    return (
      <LaunchGate
        showLicenseStep
        licenseOnly={licenseOnly}
        onAcceptLicense={handleAcceptLicense}
        onComplete={
          licenseOnly
            ? async () => {
                setSetupState((prev) => {
                  if (prev === 'loading') return prev
                  return { ...prev, needsLicense: false }
                })
              }
            : handleFirstRunComplete
        }
      />
    )
  }

  if (
    setupState.needsSetup
    && configuredBackendMode === 'managed-local'
    && !firstRunComputeConfirmed
  ) {
    return (
      <div className="h-screen overflow-auto bg-background p-6">
        <div className="mx-auto flex min-h-full w-full max-w-2xl items-center justify-center">
          <div className="w-full space-y-5">
            <div className="text-center">
              <h2 className="text-2xl font-semibold text-white">Choose where generation runs</h2>
              <p className="mx-auto mt-2 max-w-xl text-sm leading-relaxed text-zinc-400">
                LTX Desktop and your project files stay on this computer. The models can run here, or on a headless GPU machine such as a workstation or home server.
              </p>
            </div>
            <BackendConnectionPanel
              onboarding
              onContinueWithManagedLocal={() => setFirstRunComputeConfirmed(true)}
            />
          </div>
        </div>
      </div>
    )
  }

  if (setupState.needsSetup && !forceApiGenerations) {
    return <LaunchGate showLicenseStep={false} onComplete={handleFirstRunComplete} />
  }

  if (requiredModelsGate === 'missing') {
    return <LaunchGate showLicenseStep={false} onComplete={handleMissingModelsComplete} />
  }

  const renderView = () => {
    switch (currentView) {
      case 'home':
        return <Home />
      case 'project':
        return <Project />
      default:
        return <Home />
    }
  }

  return (
    <div className="relative h-screen w-screen">
      {renderView()}

      {showGlobalControls && (
        <div className="fixed top-[18px] right-3 z-50 flex items-center gap-1">
          <button
            onClick={() => {
              setSettingsInitialTab('compute')
              setIsSettingsOpen(true)
            }}
            className="flex h-8 items-center gap-1.5 rounded-md bg-black/40 px-2 text-xs text-zinc-200 ring-1 ring-white/10 backdrop-blur-sm transition-colors hover:bg-zinc-800 hover:text-white"
            title={connectionMode === 'external'
              ? `Remote backend: ${configuredBackendUrl || 'connected'}`
              : 'Models and inference run on this computer'}
          >
            {connectionMode === 'external'
              ? <Server className="h-4 w-4 text-blue-400" />
              : <Monitor className="h-4 w-4 text-blue-400" />}
            <span>{connectionMode === 'external' ? 'Remote compute' : 'This computer'}</span>
          </button>
          <button
            onClick={() => setIsLogViewerOpen(true)}
            className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
            title="View Backend Logs"
          >
            <FileText className="h-4 w-4" />
          </button>
          <button
            onClick={() => setIsSettingsOpen(true)}
            className="h-8 w-8 flex items-center justify-center rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
            title="Settings"
          >
            <Settings className="h-4 w-4" />
          </button>
        </div>
      )}

      <LogViewer isOpen={isLogViewerOpen} onClose={() => setIsLogViewerOpen(false)} />
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => {
          setIsSettingsOpen(false)
          setSettingsInitialTab(undefined)
        }}
        initialTab={settingsInitialTab}
      />
      <ApiGatewayModal
        isOpen={shouldShowGateway}
        blocking={activeApiGatewayRequest?.blocking}
        onClose={() => setApiGatewayRequest(null)}
        title={activeApiGatewayRequest?.title ?? 'Connect API Keys'}
        description={activeApiGatewayRequest?.description ?? 'Add the required API keys to continue.'}
        sections={gatewaySections}
      />
      {ltxUpgradeRecommendation && (
        <LtxUpgradePrompt
          recommendation={ltxUpgradeRecommendation}
          onClose={handleDismissLtxUpgradePrompt}
          onComplete={handleCompleteLtxUpgradePrompt}
        />
      )}

      {shouldBlockUntilSettingsLoaded && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="flex items-center gap-2 text-sm text-zinc-200">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading settings...
          </div>
        </div>
      )}

      {isForcedFirstRun && isLoaded && settings.hasLtxApiKey && isFinalizingFirstRun && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="flex items-center gap-2 text-sm text-zinc-200">
            <Loader2 className="h-4 w-4 animate-spin" />
            Finalizing setup...
          </div>
        </div>
      )}

      {isForcedFirstRun && firstRunFinalizeError && (
        <div className="fixed inset-0 z-[61] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="w-full max-w-md rounded-xl border border-zinc-700 bg-zinc-900 p-5 text-zinc-100">
            <h3 className="text-base font-semibold">Setup finalization failed</h3>
            <p className="mt-2 text-sm text-zinc-300">{firstRunFinalizeError}</p>
            <div className="mt-4 flex justify-end">
              <Button
                onClick={() => {
                  void handleFirstRunComplete().catch(() => {
                    // Error state is already captured.
                  })
                }}
              >
                Retry
              </Button>
            </div>
          </div>
        </div>
      )}

      {restartingOverlay}
      {externalBackendBanner}
    </div>
  )
}

export default function App() {
  return (
    <ProjectProvider>
      <ViewProvider>
        <KeyboardShortcutsProvider>
          <AppSettingsProvider>
            <AppContent />
            <KeyboardShortcutsModal />
          </AppSettingsProvider>
        </KeyboardShortcutsProvider>
      </ViewProvider>
    </ProjectProvider>
  )
}
