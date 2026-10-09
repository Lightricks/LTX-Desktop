import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { Loader2, AlertCircle } from 'lucide-react'
import { QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider } from 'react-router'
import { ApiClient } from './lib/api-client'
import { resolveBootPhase, resolveBootWindow, type BootModelsGate, type BootSetupState } from './lib/boot-phase'
import { isRequiredLtxBundleReady } from './lib/first-run-downloads'
import { queryClient } from './lib/queryClient'
import { ProjectProvider } from './contexts/ProjectContext'
import { KeyboardShortcutsProvider } from './contexts/KeyboardShortcutsContext'
import { AppSettingsProvider, useAppSettings } from './contexts/AppSettingsContext'
import { DevFlagsProvider } from './contexts/DevFlagsContext'
import { KeyboardShortcutsModal } from './components/KeyboardShortcutsModal'
import { useBackend } from './hooks/use-backend'
import { useGenerationRecoveryWatcher } from './hooks/use-generation-recovery-watcher'
import { logger } from './lib/logger'
import { LaunchGate } from './components/FirstRunSetup'
import { usePythonSetup } from './hooks/use-python-setup'
import { AppShellLayoutProvider } from './components/AppShell'
import { LogViewer } from './components/LogViewer'
import { StartupLoader } from './components/startup-loader/StartupLoader'
import { Button } from './components/ui/button'
import { ThemeProvider } from '@/ds/styles/themes/useTheme'
import { OptionalBootGate, wrapApp } from '@optional/app-wrap'
import { router } from './router'
import styles from './App.module.scss'

function PostSplashFade({ active, children }: { active: boolean; children: ReactNode }) {
  // The fade starts when the splash is gone, not on mount: the app boots
  // hidden underneath it, so an animation begun at mount would already have
  // run out before anyone could see it.
  return (
    <div className={`h-screen w-screen ${active ? styles.postSplashFade : ''}`}>{children}</div>
  )
}

function setDocumentWindowMode(mode: 'install' | 'app'): void {
  document.documentElement.dataset.windowMode = mode
}

function AppContent() {
  const { connected, processStatus, isLoading: backendLoading } = useBackend()
  const { forceApiGenerations, isLoaded, runtimePolicyLoaded, notifyModelsChanged } = useAppSettings()
  // Always mounted here (unlike GenSpace, which unmounts on every view/tab switch) so a
  // generation that finishes while its project isn't open still gets persisted.
  useGenerationRecoveryWatcher()

  const [pythonReady, setPythonReady] = useState<boolean | null>(null)
  const [backendStarted, setBackendStarted] = useState(false)
  const [setupState, setSetupState] = useState<BootSetupState>('loading')
  const [isFinalizingFirstRun, setIsFinalizingFirstRun] = useState(false)
  const [firstRunFinalizeError, setFirstRunFinalizeError] = useState<string | null>(null)
  const [requiredModelsGate, setRequiredModelsGate] = useState<BootModelsGate>('checking')
  const [splashFinished, setSplashFinished] = useState(false)
  const setupCompletionInFlightRef = useRef<Promise<void> | null>(null)

  const isBackendRestarting = processStatus === 'restarting'
  const isBackendDead = processStatus === 'dead'
  const waitingForRuntimePolicy = processStatus === 'alive' && !runtimePolicyLoaded

  const handleSplashFinished = useCallback(() => {
    setSplashFinished(true)
  }, [])

  // One phase drives one view (see ./lib/boot-phase.ts for the precedence).
  const phase = resolveBootPhase({
    pythonReady,
    isBackendDead,
    backendLoading,
    connected,
    waitingForRuntimePolicy,
    setupState,
    forceApiGenerations,
    requiredModelsGate,
  })
  const bootWindow = resolveBootWindow(phase, splashFinished)

  // Splash-embedded python download: unconditional hook, gated by the phase's
  // own payload (single source — the splash shows the download exactly when
  // the phase carries pythonRequired).
  const pythonActive = phase.kind === 'splash' && phase.pythonRequired

  // The endpage mix belongs to the install, so later launches start silent.
  // Null until both checks answer: the splash holds frame 0 for that beat
  // rather than starting muted and then learning it was a first launch.
  const firstLaunch =
    setupState === 'loading' || pythonReady === null
      ? null
      : setupState.needsSetup || pythonReady === false
  const {
    snapshot: pythonSnapshot,
    error: pythonError,
    retry: retryPythonDownload,
  } = usePythonSetup({ active: pythonActive })

  useLayoutEffect(() => {
    setDocumentWindowMode(bootWindow.document)
  }, [bootWindow.document])

  useEffect(() => {
    // Null while unknown or while the splash holds the window: skip the IPC.
    if (bootWindow.main === null) return
    void window.electronAPI.setMainWindowMode({ mode: bootWindow.main })
  }, [bootWindow.main])

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

  // The splash-embedded download reports progress but no completion signal,
  // so re-check readiness while python is missing; the phase advances to the
  // regular splash gates once the download lands.
  useEffect(() => {
    if (pythonReady !== false) return
    const id = window.setInterval(() => {
      const recheck = async () => {
        try {
          const result = await window.electronAPI.checkPythonReady()
          if (result.ready) setPythonReady(true)
        } catch (e) {
          logger.error(`Failed to re-check Python readiness: ${e}`)
        }
      }
      void recheck()
    }, 2000)
    return () => window.clearInterval(id)
  }, [pythonReady])

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
        setSetupState({ needsSetup: false })
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
      setSetupState({ needsSetup: false })
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

  // forceApiGenerations defaults to true until the runtime policy is read, so without
  // runtimePolicyLoaded a viable machine could auto-finalize setup and skip LaunchGate.
  const isForcedFirstRun =
    runtimePolicyLoaded &&
    setupState !== 'loading' &&
    setupState.needsSetup &&
    forceApiGenerations

  const shouldAutoFinalizeForcedFirstRun =
    isForcedFirstRun && isLoaded && !isFinalizingFirstRun && !firstRunFinalizeError

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
    return isRequiredLtxBundleReady(ltxResult.data, imgGenResult.data)
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

    if (forceApiGenerations || setupState.needsSetup) {
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

  const overlayCtx = {
    documentMode: bootWindow.document,
    mainMode: bootWindow.main,
    phaseKind: phase.kind,
    setDocumentWindowMode,
    setSplashFinished,
  }

  // One phase → one view. The shell phases (splash/setup/
  // missing-models/ready) render beneath the splash overlay, which releases
  // once the app is ready (any shell phase past 'splash').
  // Pre-ready first-run UI is pinned to the light theme its roots already
  // declare (data-theme="light"); the provider keeps Info→Tooltip portals
  // inside the themed subtree. The ready app tree stays exactly as-is.
  let beneathSplashView: ReactNode = null
  let earlyView: ReactNode = null
  switch (phase.kind) {
    case 'boot-spinner':
      earlyView = (
        <div className={styles.bootSpinner}>
          <Loader2 className="h-8 w-8 text-primary animate-spin" />
        </div>
      )
      break
    case 'backend-dead':
      earlyView = (
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
      break
    case 'splash':
      // Still loading beneath the splash: render nothing until ready.
      break
    case 'setup':
      beneathSplashView = (
        <ThemeProvider scheme="light">
          <LaunchGate onComplete={handleFirstRunComplete} />
        </ThemeProvider>
      )
      break
    case 'missing-models':
      beneathSplashView = (
        <ThemeProvider scheme="light">
          <LaunchGate onComplete={handleMissingModelsComplete} />
        </ThemeProvider>
      )
      break
    case 'ready':
      beneathSplashView = (
        <AppShellLayoutProvider
          connected={connected}
          // 'ready' implies setupState is loaded (the splash gate returns
          // otherwise); the loading arm is unreachable.
          needsSetup={setupState !== 'loading' && setupState.needsSetup}
          isBackendRestarting={isBackendRestarting}
        >
          <QueryClientProvider client={queryClient}>
            {/* Flags are fetched from the backend, so they mount with the ready app. */}
            <DevFlagsProvider>
              <RouterProvider router={router} />
            </DevFlagsProvider>
          </QueryClientProvider>

          {isForcedFirstRun && isLoaded && isFinalizingFirstRun && (
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
        </AppShellLayoutProvider>
      )
      break
    default: {
      const _exhaustive: never = phase
      throw new Error(`Unhandled boot phase: ${String(_exhaustive)}`)
    }
  }

  const content = earlyView ?? (
    <div className={styles.shell}>
      <div className={styles.beneathSplash}>
        {beneathSplashView === null ? null : (
          <PostSplashFade active={splashFinished}>{beneathSplashView}</PostSplashFade>
        )}
      </div>
      {!splashFinished && (
        <div className={styles.splashLayer}>
          <StartupLoader
            ready={phase.kind !== 'splash'}
            onFinished={handleSplashFinished}
            sound={firstLaunch}
            python={
              pythonActive
                ? {
                    snapshot: pythonSnapshot ?? {
                      statusLabel: 'Downloading Python',
                      percent: 0,
                    },
                    error: pythonError,
                    onRetry: retryPythonDownload,
                  }
                : null
            }
          />
        </div>
      )}
      {restartingOverlay}
    </div>
  )

  return <OptionalBootGate {...overlayCtx}>{content}</OptionalBootGate>
}

function App() {
  return (
    <ThemeProvider defaultPreference="system" className={styles.themeRoot}>
      <ProjectProvider>
        <KeyboardShortcutsProvider>
          <AppSettingsProvider>
            <AppContent />
            <KeyboardShortcutsModal />
          </AppSettingsProvider>
        </KeyboardShortcutsProvider>
      </ProjectProvider>
    </ThemeProvider>
  )
}

export default wrapApp(App)
