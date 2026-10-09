import { AnimatePresence } from 'framer-motion'
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Loader2 } from 'lucide-react'
import { Outlet } from 'react-router'
import { ApiClient, type ApiSuccessOf } from '../lib/api-client'
import { useAppSettings } from '../contexts/AppSettingsContext'
import { useKeyboardShortcuts } from '../contexts/KeyboardShortcutsContext'
import { logger } from '../lib/logger'
import { dismissUpgrade, isUpgradeDismissed } from '../lib/upgrade-prompt-dismissals'
import { useAppUpdateModal } from '../hooks/use-app-update'
import { useDeepLinkCourier } from '../hooks/use-deep-link-courier'
import { ApiGatewayModal, type ApiGatewaySection } from './ApiGatewayModal'
import { LogViewer } from './LogViewer'
import { LtxUpgradePrompt } from './LtxUpgradePrompt'
import type { AppUpdate } from '../hooks/use-app-update'
import {
  type SettingsOpenDetail,
  type SettingsInitialReason,
  type SettingsLegacyScrollAnchor,
  type SettingsTabId,
} from '../lib/settings-navigation'
import { SettingsScreen } from './SettingsModal'
import { UpdateAvailableModal } from './UpdateAvailableModal'
import { LtxioToastHost } from '../ltx-io/components/shared/Toast/LtxioToastHost'
import { showSuccessToast } from '../ltx-io/components/shared/Toast/toastService'

type LtxRecommendation = ApiSuccessOf<'getLtxRecommendation'>
type LtxUpgradeRecommendation = Extract<LtxRecommendation, { status: 'upgrade' }>

function isUpgradeOfferVisible(
  recommendation: LtxRecommendation,
  sessionDismissedId: string | null,
): recommendation is LtxUpgradeRecommendation {
  return recommendation.status === 'upgrade'
    && recommendation.ltx_model_id !== sessionDismissedId
    && !isUpgradeDismissed(recommendation.ltx_model_id)
}

type ApiGatewayRequest = {
  requiredKeys: Array<'ltx' | 'fal'>
  title: string
  description: string
  blocking?: boolean
  includeOptionalMissing?: boolean
}

type AppShellLayoutValue = {
  connected: boolean
  needsSetup: boolean
  isBackendRestarting: boolean
}

const AppShellLayoutContext = createContext<AppShellLayoutValue | null>(null)

type SettingsModalState = {
  isOpen: boolean
  initialTab?: SettingsTabId
  initialScrollAnchor?: SettingsLegacyScrollAnchor
  initialReason?: SettingsInitialReason
}

export type AppChromeControls = {
  openLogs: () => void
  openSettings: (detail?: SettingsOpenDetail) => void
  closeSettings: () => void
  isSettingsModalOpen: boolean
  settingsUpdate: {
    update: AppUpdate
    onOpenUpdate: () => void
    onCheckForUpdates: () => void
  }
}

const AppChromeContext = createContext<AppChromeControls | null>(null)

export function useAppChromeControls(): AppChromeControls {
  const controls = useContext(AppChromeContext)
  if (!controls) {
    throw new Error('useAppChromeControls must be used inside AppShell')
  }
  return controls
}

export function AppShellLayoutProvider({
  connected,
  needsSetup,
  isBackendRestarting,
  children,
}: AppShellLayoutValue & { children: ReactNode }) {
  const value = useMemo(
    () => ({ connected, needsSetup, isBackendRestarting }),
    [connected, needsSetup, isBackendRestarting],
  )
  return <AppShellLayoutContext.Provider value={value}>{children}</AppShellLayoutContext.Provider>
}

export function AppShell() {
  useDeepLinkCourier()
  const layout = useContext(AppShellLayoutContext) ?? {
    connected: false,
    needsSetup: true,
    isBackendRestarting: false,
  }
  const {
    settings,
    saveLtxApiKey,
    saveFalApiKey,
    forceApiGenerations,
    runtimePolicyLoaded,
    notifyModelsChanged,
  } = useAppSettings()

  const { update, isGenerationActive, isModalOpen, openModal, closeModal, checkForUpdates } = useAppUpdateModal()
  const [isLogViewerOpen, setIsLogViewerOpen] = useState(false)
  const [ltxUpgradeRecommendation, setLtxUpgradeRecommendation] = useState<LtxUpgradeRecommendation | null>(null)
  const [dismissedUpgradeTargetId, setDismissedUpgradeTargetId] = useState<LtxUpgradeRecommendation['ltx_model_id'] | null>(
    null,
  )
  const sessionDismissedUpgradeIdRef = useRef<string | null>(null)
  const [apiGatewayRequest, setApiGatewayRequest] = useState<ApiGatewayRequest | null>(null)
  const [settingsModal, setSettingsModal] = useState<SettingsModalState>({ isOpen: false })
  const { isEditorOpen: isShortcutsEditorOpen } = useKeyboardShortcuts()

  const openSettings = useCallback((detail?: SettingsOpenDetail) => {
    setSettingsModal({
      isOpen: true,
      initialTab:
        detail?.tab ??
        (detail?.reason === 'geminiKeyRequired' || detail?.reason === 'ltxKeyRequired'
          ? 'apiKeys'
          : undefined),
      initialScrollAnchor: detail?.scrollAnchor,
      initialReason: detail?.reason,
    })
  }, [])

  const closeSettings = useCallback(() => {
    setSettingsModal((prev) => ({ ...prev, isOpen: false }))
  }, [])

  const isBackendRestarting = layout.isBackendRestarting
  const waitingForRuntimePolicy = layout.connected && !runtimePolicyLoaded
  const shouldShowGateway = apiGatewayRequest !== null

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent<SettingsOpenDetail>).detail
      openSettings(detail)
    }
    window.addEventListener('open-settings', handler)
    return () => window.removeEventListener('open-settings', handler)
  }, [openSettings])

  // Platform convention: ⌘, (Ctrl+, off-Mac) opens Settings from anywhere in
  // the app. Skipped inside text entry so typing a comma never triggers it,
  // and a no-op while Settings is already open so it never yanks the tab.
  // Also skipped while the shortcuts editor is open: the capture-phase
  // listener would otherwise swallow keystrokes meant for key recording.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return
      if (!(event.metaKey || event.ctrlKey) || event.shiftKey || event.altKey) return
      if (event.code !== 'Comma' && event.key !== ',') return
      const target = event.target as HTMLElement | null
      if (target?.closest?.('input, textarea, select, [contenteditable="true"]')) return
      if (settingsModal.isOpen) return
      if (isShortcutsEditorOpen) return
      event.preventDefault()
      event.stopPropagation()
      openSettings()
    }
    window.addEventListener('keydown', onKeyDown, { capture: true })
    return () => window.removeEventListener('keydown', onKeyDown, { capture: true })
  }, [openSettings, settingsModal.isOpen, isShortcutsEditorOpen])

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

  const refreshLtxUpgradeRecommendation = useCallback(async () => {
    const result = await ApiClient.getLtxRecommendation()
    if (!result.ok) {
      logger.warn(`Failed to fetch LTX upgrade recommendation: ${result.error.message}`)
      setLtxUpgradeRecommendation(null)
      return
    }

    const recommendation = result.data
    if (isUpgradeOfferVisible(recommendation, sessionDismissedUpgradeIdRef.current)) {
      setLtxUpgradeRecommendation(recommendation)
      return
    }
    setLtxUpgradeRecommendation(null)
  }, [])

  useEffect(() => {
    if (waitingForRuntimePolicy || !layout.connected || forceApiGenerations) {
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
      if (isUpgradeOfferVisible(recommendation, sessionDismissedUpgradeIdRef.current)) {
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
    dismissedUpgradeTargetId,
    forceApiGenerations,
    layout.connected,
    waitingForRuntimePolicy,
  ])

  const hideUpgradeForSession = useCallback((modelId: LtxUpgradeRecommendation['ltx_model_id']) => {
    sessionDismissedUpgradeIdRef.current = modelId
    setDismissedUpgradeTargetId(modelId)
    setLtxUpgradeRecommendation(null)
  }, [])

  const handleDismissLtxUpgradePrompt = useCallback(() => {
    if (!ltxUpgradeRecommendation) return
    dismissUpgrade(ltxUpgradeRecommendation.ltx_model_id)
    hideUpgradeForSession(ltxUpgradeRecommendation.ltx_model_id)
  }, [hideUpgradeForSession, ltxUpgradeRecommendation])

  const handleCompleteLtxUpgradePrompt = useCallback(async () => {
    // A finished download is not a dismissal. Hiding it for this session keeps
    // the prompt from flashing back if the active model has not flipped yet.
    // If that checkpoint is removed later, the prompt can return.
    const modelId = ltxUpgradeRecommendation?.ltx_model_id
    if (modelId) hideUpgradeForSession(modelId)
    showSuccessToast(modelId ? `${modelId} weights downloaded` : 'New model weights downloaded')
    notifyModelsChanged()
    await refreshLtxUpgradeRecommendation()
  }, [hideUpgradeForSession, ltxUpgradeRecommendation, notifyModelsChanged, refreshLtxUpgradeRecommendation])

  const gatewaySections: ApiGatewaySection[] = useMemo(() => {
    if (!apiGatewayRequest) return []

    const sections: ApiGatewaySection[] = [
      {
        keyType: 'ltx',
        title: 'LTX API',
        description: 'Video generation, prompt enhancement, and cloud text encoding.',
        required: apiGatewayRequest.requiredKeys.includes('ltx'),
        isConfigured: settings.hasLtxApiKey,
        inputLabel: 'LTX API key',
        placeholder: 'Enter your LTX API key...',
        onSave: saveLtxApiKey,
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
    saveFalApiKey,
    saveLtxApiKey,
    settings.hasFalApiKey,
    settings.hasLtxApiKey,
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

  const chromeControls = useMemo<AppChromeControls>(
    () => ({
      openLogs: () => setIsLogViewerOpen(true),
      openSettings,
      closeSettings,
      isSettingsModalOpen: settingsModal.isOpen,
      settingsUpdate: {
        update,
        onOpenUpdate: openModal,
        onCheckForUpdates: checkForUpdates,
      },
    }),
    [checkForUpdates, closeSettings, openModal, openSettings, settingsModal.isOpen, update],
  )

  return (
    <AppChromeContext.Provider value={chromeControls}>
    <div className="relative h-screen w-screen">
      <Outlet />

      <LogViewer isOpen={isLogViewerOpen} onClose={() => setIsLogViewerOpen(false)} />
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

      <AnimatePresence>
        {settingsModal.isOpen ? (
          <SettingsScreen
            key="settings-modal"
            isOpen={settingsModal.isOpen}
            onClose={closeSettings}
            initialTab={settingsModal.initialTab}
            initialScrollAnchor={settingsModal.initialScrollAnchor}
            initialReason={settingsModal.initialReason}
            update={update}
            onOpenUpdate={openModal}
            onCheckForUpdates={checkForUpdates}
          />
        ) : null}
      </AnimatePresence>

      {restartingOverlay}
      <LtxioToastHost />
    </div>
    </AppChromeContext.Provider>
  )
}
