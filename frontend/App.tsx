import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Loader2, AlertCircle, Settings, FileText } from 'lucide-react'
import { ApiClient, type ApiSuccessOf } from './lib/api-client'
import { ProjectProvider } from './contexts/ProjectContext'
import { ViewProvider, useView } from './contexts/ViewContext'
import { KeyboardShortcutsProvider, useKeyboardShortcuts } from './contexts/KeyboardShortcutsContext'
import { AppSettingsProvider, useAppSettings } from './contexts/AppSettingsContext'
import { DevFlagsProvider } from './contexts/DevFlagsContext'
import { KeyboardShortcutsModal } from './components/KeyboardShortcutsModal'
import { DevPanel } from './components/DevPanel'
import { useBackend } from './hooks/use-backend'
import { useGenerationRecoveryWatcher } from './hooks/use-generation-recovery-watcher'
import { logger } from './lib/logger'
import { Home } from './views/Home'
import { Project } from './views/Project'
import { LaunchGate } from './components/FirstRunSetup'
import { LtxUpgradePrompt } from './components/LtxUpgradePrompt'
import { dismissUpgrade, isUpgradeDismissed } from './lib/upgrade-prompt-dismissals'
import { PythonSetup } from './components/PythonSetup'
import { SettingsModal, type SettingsInitialReason, type SettingsTabId } from './components/SettingsModal'
import { LogViewer } from './components/LogViewer'
import { ApiGatewayModal, type ApiGatewaySection } from './components/ApiGatewayModal'
import { Button } from './components/ui/button'
import { PromptManagerPro } from './components/gpm/PromptManagerPro'
import { DownloadsBrowser } from './components/gpm/DownloadsBrowser'
import { useDownloadsBrowserOpen, getDownloadsBrowserOpen, setDownloadsBrowserOpen } from './components/gpm/downloads-browser-store'
import { usePromptManagerProOpen, getPromptManagerProOpen, setPromptManagerProOpen } from './components/gpm/prompt-manager-pro-store'
import { useAppUpdateModal } from './hooks/use-app-update'
import { UpdateAvailableModal } from './components/UpdateAvailableModal'
import { dispatchMcpEditorRequest } from './views/editor/mcp-editor-tools'

type SetupState = 'loading' | { needsSetup: boolean; needsLicense: boolean }
type RequiredModelsGateState = 'checking' | 'missing' | 'ready'
type LtxRecommendation = ApiSuccessOf<'getLtxRecommendation'>
type LtxUpgradeRecommendation = Extract<LtxRecommendation, { status: 'upgrade' }>

function AppContent() {
  const { currentView } = useView()
  const downloadsBrowserOpen = useDownloadsBrowserOpen()
  const promptManagerProOpen = usePromptManagerProOpen()
  const { setEditorOpen: setKbEditorOpen } = useKeyboardShortcuts()

  // Tab toggles both Prompt Manager Pro panels closed/open together — like
  // Photoshop/Premiere's "hide all panels" — restoring whichever side(s) were
  // actually open before collapsing.
  const panelsCollapsedRef = useRef(false)
  const savedPanelStateRef = useRef({ left: false, right: true })
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return
      const el = document.activeElement as HTMLElement | null
      const isEditable = !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable)
      if (isEditable) return
      e.preventDefault()
      if (!panelsCollapsedRef.current) {
        savedPanelStateRef.current = { left: getDownloadsBrowserOpen(), right: getPromptManagerProOpen() }
        setDownloadsBrowserOpen(false)
        setPromptManagerProOpen(false)
        panelsCollapsedRef.current = true
      } else {
        setDownloadsBrowserOpen(savedPanelStateRef.current.left)
        setPromptManagerProOpen(savedPanelStateRef.current.right)
        panelsCollapsedRef.current = false
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])
  // RiX MCP server: answer editor tool calls forwarded from the main process. Lives
  // here (always mounted) so a call with no editor open fails fast instead of timing out.
  useEffect(() => window.electronAPI?.onMcpEditorRequest?.(({ id, tool, args }) => {
    void dispatchMcpEditorRequest(tool, args)
      .then(result => window.electronAPI.mcpEditorResponse({ id, ok: true, result }))
      .catch((e: unknown) => window.electronAPI.mcpEditorResponse({ id, ok: false, error: e instanceof Error ? e.message : String(e) }))
  }), [])
  const { connected, processStatus, isLoading: backendLoading } = useBackend()
  const { settings, saveLtxApiKey, saveFalApiKey, forceApiGenerations, isLoaded, runtimePolicyLoaded, notifyModelsChanged } = useAppSettings()
  // Always mounted here (unlike GenSpace, which unmounts on every view/tab switch) so a
  // generation that finishes while its project isn't open still gets persisted.
  useGenerationRecoveryWatcher()

  const [pythonReady, setPythonReady] = useState<boolean | null>(null)
  const [backendStarted, setBackendStarted] = useState(false)
  const [setupState, setSetupState] = useState<SetupState>('loading')
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [settingsInitialTab, setSettingsInitialTab] = useState<SettingsTabId | undefined>(undefined)
  const [settingsInitialReason, setSettingsInitialReason] = useState<SettingsInitialReason | undefined>(undefined)
  const { update, isGenerationActive, isModalOpen, openModal, closeModal, checkForUpdates } = useAppUpdateModal()
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

  const isBackendRestarting = processStatus === 'restarting'
  const isBackendDead = processStatus === 'dead'
  const waitingForRuntimePolicy = processStatus === 'alive' && !runtimePolicyLoaded

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail
      if (detail?.tab) setSettingsInitialTab(detail.tab)
      setSettingsInitialReason(detail?.reason === 'geminiKeyRequired' ? 'geminiKeyRequired' : undefined)
      setIsSettingsOpen(true)
    }
    window.addEventListener('open-settings', handler)
    return () => window.removeEventListener('open-settings', handler)
  }, [])

  // Forward the native File menu's Project actions (Save/Export/Import/New)
  // into the renderer as a window event, the same way other cross-component
  // signals here are handled — Home.tsx/Project.tsx listen for the actions
  // relevant to whichever of them is currently mounted.
  useEffect(() => {
    return window.electronAPI?.onMenuAction((action) => {
      if (action === 'show-keyboard-shortcuts') {
        setKbEditorOpen(true)
        return
      }
      window.dispatchEvent(new CustomEvent('ltx:menu-action', { detail: action }))
    })
  }, [setKbEditorOpen])

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
        const result = await window.electronAPI.checkPythonReady()
        setPythonReady(result.ready)
      } catch (e) {
        logger.error(`Failed to check Python readiness: ${e}`)
        setPythonReady(true)
      }
    }
    void check()
  }, [])

  useEffect(() => {
    if (pythonReady !== true || backendStarted) return
    setBackendStarted(true)
    const start = async () => {
      try {
        logger.info('Starting Python backend...')
        await window.electronAPI.startPythonBackend()
        logger.info('Python backend started successfully')
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
      notifyModelsChanged()
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
  }, [notifyModelsChanged])

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

    // With an LTX API key, generation can always run via the (free) LTX API, so
    // never block startup on missing local models — local downloads stay
    // available in Settings. (Also avoids a cold-boot model-scan race wrongly
    // demanding the local text encoder.)
    if (forceApiGenerations || settings.hasLtxApiKey || setupState.needsLicense || setupState.needsSetup) {
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
    settings.hasLtxApiKey,
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
    if (recommendation.status === 'upgrade' && recommendation.ltx_model_id !== dismissedUpgradeTargetId && !isUpgradeDismissed(recommendation.ltx_model_id)) {
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
      if (recommendation.status === 'upgrade' && recommendation.ltx_model_id !== dismissedUpgradeTargetId && !isUpgradeDismissed(recommendation.ltx_model_id)) {
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

  const handleDontShowLtxUpgradeAgain = useCallback(() => {
    if (!ltxUpgradeRecommendation) return
    dismissUpgrade(ltxUpgradeRecommendation.ltx_model_id)  // persist for this model id
    setDismissedUpgradeTargetId(ltxUpgradeRecommendation.ltx_model_id)
    setLtxUpgradeRecommendation(null)
  }, [ltxUpgradeRecommendation])

  const handleCompleteLtxUpgradePrompt = useCallback(async () => {
    setDismissedUpgradeTargetId(null)
    notifyModelsChanged()
    await refreshLtxUpgradeRecommendation()
  }, [notifyModelsChanged, refreshLtxUpgradeRecommendation])

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

  const showGlobalControls = currentView !== 'home' && connected && setupState !== 'loading' && !setupState.needsSetup
  const shouldBlockUntilSettingsLoaded = forceApiGenerations && !isLoaded
  const shouldShowForcedFirstRunUpsell = isForcedFirstRun && isLoaded && !settings.hasLtxApiKey
  const shouldShowGlobalForcedUpsell = forceApiGenerations && setupState !== 'loading' && !setupState.needsSetup && isLoaded && !settings.hasLtxApiKey
  const shouldBlockForLtxKey = shouldShowForcedFirstRunUpsell || shouldShowGlobalForcedUpsell

  useEffect(() => {
    if (shouldBlockForLtxKey && apiGatewayRequest === null) {
      setApiGatewayRequest({
        requiredKeys: ['ltx'],
        title: 'Connect API Keys',
        description: 'This app is configured for API-only generation. Add your API key to continue.',
        blocking: true,
        includeOptionalMissing: true,
      })
    }
  }, [shouldBlockForLtxKey, apiGatewayRequest])

  const shouldShowGateway = apiGatewayRequest !== null

  const gatewaySections: ApiGatewaySection[] = useMemo(() => {
    if (!apiGatewayRequest) return []

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
        title: 'LTX API',
        description: 'Video generation, prompt enhancement, and cloud text encoding.',
        required: apiGatewayRequest.requiredKeys.includes('ltx'),
        isConfigured: settings.hasLtxApiKey,
        inputLabel: 'LTX API key',
        placeholder: 'Enter your LTX API key...',
        onSave: handleSaveLtxKey,
        onGetKey: () => window.electronAPI.openLtxApiKeyPage(),
        getKeyLabel: 'Get LTX API key',
      },
      {
        keyType: 'fal',
        title: 'FAL AI',
        description: 'Required to generate or edit images with Z Image Turbo.',
        required: apiGatewayRequest.requiredKeys.includes('fal'),
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
      if (apiGatewayRequest.includeOptionalMissing) return true
      return false
    })
  }, [
    apiGatewayRequest,
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
    return <PythonSetup onReady={() => setPythonReady(true)} />
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
            <h2 className="text-xl font-semibold text-foreground mb-2">Starting RiX Desktop Studio Pro...</h2>
            <p className="text-muted-foreground">Initializing the inference engine</p>
          </div>
        </div>
        {restartingOverlay}
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
      <div
        className="h-full transition-[margin-left,width] duration-150"
        style={{
          marginLeft: downloadsBrowserOpen ? 340 : 0,
          width: `calc(100% - ${downloadsBrowserOpen ? 340 : 0}px - ${promptManagerProOpen ? 396 : 0}px)`,
        }}
      >
        {renderView()}
      </div>

      {showGlobalControls && (
        <div className="fixed top-[18px] right-3 z-50 flex items-center gap-1">
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
          setSettingsInitialReason(undefined)
        }}
        initialTab={settingsInitialTab}
        initialReason={settingsInitialReason}
        update={update}
        onOpenUpdate={openModal}
        onCheckForUpdates={checkForUpdates}
      />
      <ApiGatewayModal
        isOpen={shouldShowGateway}
        blocking={apiGatewayRequest?.blocking}
        onClose={() => setApiGatewayRequest(null)}
        title={apiGatewayRequest?.title ?? 'Connect API Keys'}
        description={apiGatewayRequest?.description ?? 'Add the required API keys to continue.'}
        sections={gatewaySections}
      />
      {ltxUpgradeRecommendation && (
        <LtxUpgradePrompt
          recommendation={ltxUpgradeRecommendation}
          onClose={handleDismissLtxUpgradePrompt}
          onDontShowAgain={handleDontShowLtxUpgradeAgain}
          onComplete={handleCompleteLtxUpgradePrompt}
        />
      )}
      {isModalOpen && (
        <UpdateAvailableModal
          update={update}
          isGenerationActive={isGenerationActive}
          onClose={closeModal}
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

      <PromptManagerPro />
      <DownloadsBrowser />
    </div>
  )
}

export default function App() {
  return (
    <ProjectProvider>
      <ViewProvider>
        <KeyboardShortcutsProvider>
          <AppSettingsProvider>
            <DevFlagsProvider>
              <AppContent />
              <KeyboardShortcutsModal />
              <DevPanel />
            </DevFlagsProvider>
          </AppSettingsProvider>
        </KeyboardShortcutsProvider>
      </ViewProvider>
    </ProjectProvider>
  )
}
