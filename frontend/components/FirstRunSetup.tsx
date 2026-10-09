import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@ds/Button/Button'
import { Text } from '@ds/Text/Text'
import { TooltipProvider } from '@ds/Tooltip/Tooltip'

import { useAppSettings } from '@/contexts/AppSettingsContext'
import { useHfAuth } from '@/hooks/use-hf-auth'
import { useHfModelAccess } from '@/hooks/use-hf-model-access'
import { ApiClient, type ApiRequestBodyOf, type ApiSuccessOf } from '@/lib/api-client'
import {
  buildFirstRunDownloadSteps,
  defaultSelectedFirstRunLoraKeys,
  defaultSelectedRecommendedQualityCpIds,
  describeCheckpointsErrorMessage,
  firstRunCatalogCpIds,
  firstRunLoraFromCatalog,
  firstRunLoraKey,
  isFirstRunImageModel,
  isFirstRunModelPackItem,
  isFirstRunTextEncoder,
  recommendationsErrorMessage,
  recommendedQualityCpIds,
  uniqueCpIds,
  type FirstRunDownloadStep,
  type FirstRunLoraItem,
} from '@/lib/first-run-downloads'
import { formatBytes } from '@/lib/format'
import { isHomeFeatureEnabled } from '@/lib/home-features'
import { logger } from '@/lib/logger'

import { HfAccessCard } from './first-run/HfAccessCard'
import { LtxApiKeyCard } from './first-run/LtxApiKeyCard'
import { ModelDownloadsCard } from './first-run/ModelDownloadsCard'
import { flushAnalyticsOptOut, SetupAnalyticsOptOut } from './first-run/SetupAnalyticsOptOut'
import { SetupInstallingStep } from './first-run/SetupInstallingStep'
import { SetupLegalDisclaimer } from './first-run/SetupLegalDisclaimer'
import { useDownloadQueue, type DownloadQueueItem } from './first-run/useDownloadQueue'
import styles from './FirstRunSetup.module.scss'

interface LaunchGateProps {
  onComplete: () => Promise<void>
}

type Step = 'location' | 'installing'
type StartModelDownloadBody = NonNullable<ApiRequestBodyOf<'startModelDownload'>>
type ModelCheckpointID = NonNullable<StartModelDownloadBody['cp_ids']>[number]
type LtxRecommendation = ApiSuccessOf<'getLtxRecommendation'>
type ImgGenRecommendation = ApiSuccessOf<'getImgGenRecommendation'>
type CheckpointDescriptor = ApiSuccessOf<'describeCheckpoints'>['checkpoints'][number]

function buildDownloadSteps(
  ltxRecommendation: LtxRecommendation,
  imgGenRecommendation: ImgGenRecommendation,
  extraCpIds: readonly ModelCheckpointID[] = [],
  selectedRecommendedQualityCpIds: readonly ModelCheckpointID[] = [],
  excludedCpIds: readonly ModelCheckpointID[] = [],
  includeImageModel = true,
): FirstRunDownloadStep<ModelCheckpointID>[] {
  return buildFirstRunDownloadSteps({
    ltxRecommendation,
    imgGenCpToDownload: includeImageModel ? imgGenRecommendation.cp_to_download : null,
    selectedRecommendedQualityCpIds,
    optedInOptionalCpIds: extraCpIds,
    excludedCpIds,
  })
}

export function LaunchGate({ onComplete }: LaunchGateProps) {
  const [currentStep, setCurrentStep] = useState<Step>('location')
  const [installPath, setInstallPath] = useState('')
  const [installFinished, setInstallFinished] = useState(false)
  const [availableSpace, setAvailableSpace] = useState('...')
  const [recommendationsError, setRecommendationsError] = useState<string | null>(null)
  const [downloadItems, setDownloadItems] = useState<CheckpointDescriptor[]>([])
  const [optionalItems, setOptionalItems] = useState<CheckpointDescriptor[]>([])
  const [recommendedItems, setRecommendedItems] = useState<CheckpointDescriptor[]>([])
  const [selectedRecommendedCpIds, setSelectedRecommendedCpIds] = useState<ModelCheckpointID[]>([])
  const [loraItems, setLoraItems] = useState<FirstRunLoraItem[]>([])
  const [selectedLoraKeys, setSelectedLoraKeys] = useState<string[]>([])
  const [downloadTextEncoder, setDownloadTextEncoder] = useState(true)
  const [downloadImageModel, setDownloadImageModel] = useState(true)
  const [ltxApiKey, setLtxApiKey] = useState('')
  const [hasSavedLtxApiKey, setHasSavedLtxApiKey] = useState(false)
  const [apiKeyError, setApiKeyError] = useState<string | null>(null)
  const [replacingApiKey, setReplacingApiKey] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const finishTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const completeSetupNow = useCallback(async () => {
    setActionError(null)
    try {
      await onComplete()
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Failed to complete setup.')
    }
  }, [onComplete])

  useEffect(() => {
    return () => {
      if (finishTimeoutRef.current != null) window.clearTimeout(finishTimeoutRef.current)
    }
  }, [])

  const queue = useDownloadQueue({
    enabled: currentStep === 'installing',
    onComplete: () => {
      setInstallFinished(true)
      if (finishTimeoutRef.current != null) window.clearTimeout(finishTimeoutRef.current)
      finishTimeoutRef.current = setTimeout(() => {
        finishTimeoutRef.current = null
        void completeSetupNow()
      }, 600)
    },
  })
  const liveHfAuth = useHfAuth(currentStep === 'location')
  const apiKeyCollapsed = hasSavedLtxApiKey && !replacingApiKey
  const encoderItem = useMemo(
    () =>
      downloadItems.find(isFirstRunTextEncoder) ?? optionalItems.find(isFirstRunTextEncoder) ?? null,
    [downloadItems, optionalItems],
  )
  const encoderOnDisk = Boolean(encoderItem?.downloaded)
  const encoderSelected = encoderOnDisk || downloadTextEncoder
  const showApiKeyCard = Boolean(encoderItem) && !encoderSelected
  const [apiKeyMotionReady, setApiKeyMotionReady] = useState(false)

  useEffect(() => {
    if (currentStep !== 'location') {
      setApiKeyMotionReady(false)
      return
    }
    let inner = 0
    const outer = requestAnimationFrame(() => {
      inner = requestAnimationFrame(() => setApiKeyMotionReady(true))
    })
    return () => {
      cancelAnimationFrame(outer)
      cancelAnimationFrame(inner)
    }
  }, [currentStep])
  const modelPackItems = useMemo(() => downloadItems.filter(isFirstRunModelPackItem), [downloadItems])
  const imageItem = useMemo(() => downloadItems.find(isFirstRunImageModel) ?? null, [downloadItems])
  const imageOnDisk = Boolean(imageItem?.downloaded)
  const imageSelected = imageOnDisk || downloadImageModel
  const isSelectedRecommended = (item: CheckpointDescriptor) =>
    selectedRecommendedCpIds.includes(item.cp_id)
  const toggleRecommended = (item: CheckpointDescriptor) => {
    const cpId = item.cp_id
    setSelectedRecommendedCpIds((ids) =>
      ids.includes(cpId) ? ids.filter((id) => id !== cpId) : [...ids, cpId],
    )
  }
  const toggleLora = (key: string) => {
    setSelectedLoraKeys((keys) =>
      keys.includes(key) ? keys.filter((itemKey) => itemKey !== key) : [...keys, key],
    )
  }
  const selectableLoraKeys = loraItems.filter((item) => !item.downloaded).map(firstRunLoraKey)
  const selectedSelectableLoraCount = selectableLoraKeys.filter((key) =>
    selectedLoraKeys.includes(key),
  ).length
  const allLorasSelected =
    selectableLoraKeys.length > 0 && selectedSelectableLoraCount === selectableLoraKeys.length
  const lorasIndeterminate =
    selectedSelectableLoraCount > 0 && selectedSelectableLoraCount < selectableLoraKeys.length
  const toggleAllLoras = () => {
    setSelectedLoraKeys(allLorasSelected ? [] : selectableLoraKeys)
  }
  const pendingLoraItems = useMemo(
    () => loraItems.filter((item) => !item.downloaded && selectedLoraKeys.includes(firstRunLoraKey(item))),
    [loraItems, selectedLoraKeys],
  )
  const pendingDownloadItems = useMemo(
    () =>
      [
        ...modelPackItems,
        ...(encoderSelected && encoderItem ? [encoderItem] : []),
        ...(imageSelected && imageItem ? [imageItem] : []),
        ...recommendedItems.filter((item) => selectedRecommendedCpIds.includes(item.cp_id)),
      ].filter((item) => !item.downloaded),
    [
      modelPackItems,
      encoderSelected,
      encoderItem,
      imageSelected,
      imageItem,
      recommendedItems,
      selectedRecommendedCpIds,
    ],
  )
  const installCpIds = useMemo(
    () => uniqueCpIds(pendingDownloadItems.map((item) => item.cp_id)),
    [pendingDownloadItems],
  )
  const liveAccess = useHfModelAccess(
    currentStep === 'location' ? installCpIds : [],
    liveHfAuth.hfAuthStatus,
  )
  const hfAuthStatus = liveHfAuth.hfAuthStatus
  const hfAuthPolling = liveHfAuth.hfAuthPolling
  const startHuggingFaceLogin = liveHfAuth.startHuggingFaceLogin
  const handleHuggingFaceLogout = liveHfAuth.handleHuggingFaceLogout
  const accessMap = liveAccess.accessMap
  const allAuthorized = liveAccess.allAuthorized
  const checkingAccess = liveAccess.checking
  const checkError = liveAccess.checkError
  const recheckAccess = liveAccess.recheckAccess
  const hfUnsigned = hfAuthStatus !== 'authenticated'
  const hfNeedsLicense = !hfUnsigned && !allAuthorized && !checkError
  const hfComplete = !hfUnsigned && allAuthorized
  const hfShowGate = Boolean(checkError) || hfNeedsLicense
  const { saveLtxApiKey } = useAppSettings()
  const downloadProgress = queue.progress
  const downloadError = queue.error
  const runningDownloadProgress = downloadProgress?.status === 'downloading' ? downloadProgress : null
  const totalProgress =
    runningDownloadProgress?.total_progress ?? (downloadProgress?.status === 'complete' ? 100 : 0)

  const totalDownloadBytes =
    pendingDownloadItems.reduce((sum, item) => sum + item.size_bytes, 0) +
    pendingLoraItems.reduce((sum, item) => sum + item.sizeBytes, 0)

  const refreshModelRecommendations = useCallback(async () => {
    setRecommendationsError(null)

    const [settingsResult, ltxResult, imgGenResult, loraResult, icLoraResult] = await Promise.all([
      ApiClient.getSettings(),
      ApiClient.getLtxRecommendation({ include_installed: true }),
      ApiClient.getImgGenRecommendation({ fresh: true }),
      ApiClient.listLoras({ fresh: true }),
      ApiClient.listIcLoras({ fresh: true }),
    ])
    if (!settingsResult.ok) {
      const message = recommendationsErrorMessage(settingsResult.error.message)
      logger.error(message)
      setRecommendationsError(message)
      return
    }
    if (!ltxResult.ok) {
      const message = recommendationsErrorMessage(ltxResult.error.message)
      logger.error(message)
      setRecommendationsError(message)
      return
    }
    if (!imgGenResult.ok) {
      const message = recommendationsErrorMessage(imgGenResult.error.message)
      logger.error(message)
      setRecommendationsError(message)
      return
    }

    setInstallPath(settingsResult.data.modelsDir ?? '')
    setHasSavedLtxApiKey(Boolean(settingsResult.data.hasLtxApiKey))
    const nextLoraItems = [
      ...(loraResult.ok
        ? loraResult.data.loras.map((entry) => firstRunLoraFromCatalog('lora', entry.lora, entry.downloaded))
        : []),
      ...(icLoraResult.ok
        ? icLoraResult.data.ic_loras
            .filter((entry) => isHomeFeatureEnabled(entry.ic_lora.id))
            .map((entry) => firstRunLoraFromCatalog('ic-lora', entry.ic_lora, entry.downloaded))
        : []),
    ]
    setLoraItems(nextLoraItems)
    setSelectedLoraKeys(defaultSelectedFirstRunLoraKeys(nextLoraItems))

    // Full sets for the model Install will download, including files already on
    // disk. `include_installed` stays on that model (an older incomplete bundle
    // is not replaced by the latest one). Image `fresh` only keeps the single
    // image checkpoint visible once it is downloaded. describe marks on-disk
    // files. Install still re-fetches the live missing-only plan.
    const recommendedCpIds = recommendedQualityCpIds(ltxResult.data)
    const optionalCpIds = ltxResult.data.status === 'download' ? ltxResult.data.optional_cp_ids : []
    const catalogCpIds = firstRunCatalogCpIds({
      ltxRecommendation: ltxResult.data,
      imgGenCpToDownload: imgGenResult.data.cp_to_download,
    })
    if (catalogCpIds.length === 0) {
      setDownloadItems([])
      setOptionalItems([])
      setRecommendedItems([])
      setSelectedRecommendedCpIds([])
      return
    }
    const describeResult = await ApiClient.describeCheckpoints({
      cp_ids: catalogCpIds,
    })
    if (!describeResult.ok) {
      const message = describeCheckpointsErrorMessage(describeResult.error.message)
      logger.error(message)
      setRecommendationsError(message)
      return
    }
    const described = describeResult.data.checkpoints
    const isOptional = (item: CheckpointDescriptor) => optionalCpIds.includes(item.cp_id)
    const isRecommended = (item: CheckpointDescriptor) => recommendedCpIds.includes(item.cp_id)
    const downloadedCpIds = new Set(described.filter((item) => item.downloaded).map((item) => item.cp_id))
    setDownloadItems(described.filter((item) => !isOptional(item) && !isRecommended(item)))
    setOptionalItems(described.filter(isOptional))
    setRecommendedItems(described.filter(isRecommended))
    setSelectedRecommendedCpIds(
      defaultSelectedRecommendedQualityCpIds(ltxResult.data).filter((cpId) => !downloadedCpIds.has(cpId)),
    )
  }, [])

  // Initialize
  useEffect(() => {
    const init = async () => {
      try {
        await refreshModelRecommendations()
      } catch (e) {
        logger.error(`Init error: ${e}`)
      }
    }
    init()
  }, [refreshModelRecommendations])

  useEffect(() => {
    if (!installPath) {
      setAvailableSpace('...')
      return
    }
    let cancelled = false
    void (async () => {
      try {
        const result = await window.electronAPI.getFreeDiskSpace({ path: installPath })
        if (cancelled) return
        setAvailableSpace(result.success ? formatBytes(result.bytes) : '—')
      } catch {
        if (!cancelled) setAvailableSpace('—')
      }
    })()
    return () => {
      cancelled = true
    }
  }, [installPath])

  // LTX is stored as typed, with no provider check. A bad key fails on the first
  // generation, which opens Settings on this field. Presence gating stays
  // (skipping the local text model without any key blocks Install).
  // Returns false only when the user has to fix something.
  const persistApiKeyIfNeeded = async (requireKey: boolean): Promise<boolean> => {
    if (!ltxApiKey) {
      if (requireKey && !hasSavedLtxApiKey) {
        setApiKeyError('Enter an API key.')
        return false
      }
      return true
    }
    setApiKeyError(null)
    try {
      await saveLtxApiKey(ltxApiKey)
      setHasSavedLtxApiKey(true)
      setReplacingApiKey(false)
      setLtxApiKey('')
      return true
    } catch (e) {
      logger.error(`Failed to save API key: ${e instanceof Error ? e.message : String(e)}`)
      setApiKeyError(
        e instanceof Error && e.message
          ? e.message
          : 'Couldn’t save this key. Please try again.',
      )
      return false
    }
  }

  // Start installation
  const startInstallation = async () => {
    const requireKey = !encoderSelected
    if (!(await persistApiKeyIfNeeded(requireKey))) return
    // The opt-out checkbox writes on toggle. Wait for that write so `launched`
    // sees the choice the user just made. A failed write must not send the event
    // while analytics are still enabled.
    try {
      await flushAnalyticsOptOut()
    } catch (e) {
      setActionError(
        e instanceof Error && e.message
          ? e.message
          : 'Couldn’t save the analytics choice. Please try again.',
      )
      return
    }
    await window.electronAPI.notifyInstallStarted().catch(() => {})
    setCurrentStep('installing')
    setInstallFinished(false)
    try {
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
      const nextLtxRecommendation = ltxResult.data
      const nextImgGenRecommendation = imgGenResult.data
      const encoderCpId = encoderItem?.cp_id
      const imageCpId = imageItem?.cp_id
      const extraCpIds = encoderSelected && encoderCpId && !encoderOnDisk ? [encoderCpId] : []
      const excludedCpIds = [
        ...(!encoderSelected && encoderCpId ? [encoderCpId] : []),
        ...(!imageSelected && imageCpId ? [imageCpId] : []),
      ]

      const downloadSteps = buildDownloadSteps(
        nextLtxRecommendation,
        nextImgGenRecommendation,
        extraCpIds,
        selectedRecommendedCpIds,
        excludedCpIds,
        imageSelected,
      )
      const items: DownloadQueueItem[] = [
        ...downloadSteps.map((step): DownloadQueueItem => ({ kind: 'model-step', step })),
        ...pendingLoraItems.map((item): DownloadQueueItem => ({ kind: 'lora', item })),
      ]
      await queue.start(items)
    } catch (e) {
      logger.error(`Download start error: ${e}`)
      queue.fail(e instanceof Error ? e.message : 'Failed to start model download.')
    }
  }

  const retryInstallation = () => {
    queue.reset()
    void startInstallation()
  }

  const handleChangeModelsDir = useCallback(() => {
    void (async () => {
      const result = await window.electronAPI?.openModelsDirChangeDialog()
      if (result?.success) setInstallPath(result.path)
    })()
  }, [])

  // Handle next button
  const handleNext = async () => {
    setActionError(null)
    if (currentStep === 'location') {
      await startInstallation()
      return
    }
  }

  // Get button text
  const getNextButtonText = () => {
    if (currentStep === 'location') return 'Install'
    return 'Continue'
  }

  // Check if next button should be disabled
  const isNextDisabled = () => {
    if (currentStep === 'location') {
      const apiKeyMissing = showApiKeyCard && !hasSavedLtxApiKey && ltxApiKey.length === 0
      return (installCpIds.length > 0 && (!allAuthorized || checkingAccess)) || apiKeyMissing
    }
    return false
  }

  return (
    <TooltipProvider>
      <div
        className={`${styles.root} ${currentStep === 'installing' ? styles.rootFlush : ''}`}
      >
        <div className={styles.dragRegion} aria-hidden />
        <div className={styles.main}>
          <div className={`${styles.content} ${currentStep === 'installing' ? styles.contentFlush : ''}`}>
            {currentStep === 'location' && (
              <div className={`${styles.step} ${styles.stepScroll}`}>
                <Text as="h2" variant="heading" size="xl" className={`${styles.stepTitle} ${styles.stepTitleOnly}`}>
                  Setup
                </Text>

                <div
                  className={`${styles.accountRow} ${showApiKeyCard ? styles.accountRowSplit : ''} ${apiKeyMotionReady ? styles.accountRowReady : ''}`}
                >
                  <HfAccessCard
                    complete={hfComplete}
                    showGate={hfShowGate}
                    accessMap={accessMap}
                    allAuthorized={allAuthorized}
                    hfAuthStatus={hfAuthStatus}
                    hfAuthPolling={hfAuthPolling}
                    unsigned={hfUnsigned}
                    checkError={checkError}
                    onSignIn={startHuggingFaceLogin}
                    onSignOut={handleHuggingFaceLogout}
                    onRetryCheck={recheckAccess}
                  />

                  {encoderItem && (
                    <div
                      className={styles.apiKeyPane}
                      aria-hidden={!showApiKeyCard}
                      // `inert` is missing from React 18's types, so spread it to dodge excess-prop checking.
                      {...(!showApiKeyCard ? { inert: '' } : {})}
                    >
                      <LtxApiKeyCard
                        collapsed={apiKeyCollapsed}
                        hasSavedKey={hasSavedLtxApiKey}
                        apiKey={ltxApiKey}
                        apiKeyError={apiKeyError}
                        onApiKeyChange={(value) => {
                          setLtxApiKey(value)
                          setApiKeyError(null)
                        }}
                        onChangeKey={() => setReplacingApiKey(true)}
                      />
                    </div>
                  )}
                </div>

                <ModelDownloadsCard
                  modelPackItems={modelPackItems}
                  encoderItem={encoderItem}
                  encoderSelected={encoderSelected}
                  onToggleEncoder={() => setDownloadTextEncoder((value) => !value)}
                  recommendedItems={recommendedItems}
                  isRecommendedSelected={isSelectedRecommended}
                  onToggleRecommended={toggleRecommended}
                  imageItem={imageItem}
                  imageSelected={imageSelected}
                  onToggleImage={() => setDownloadImageModel((value) => !value)}
                  loraItems={loraItems}
                  selectedLoraKeys={selectedLoraKeys}
                  onToggleLora={toggleLora}
                  allLorasSelected={allLorasSelected}
                  lorasIndeterminate={lorasIndeterminate}
                  onToggleAllLoras={toggleAllLoras}
                  pendingLoraItems={pendingLoraItems}
                  totalDownloadBytes={totalDownloadBytes}
                  recommendationsError={recommendationsError}
                  onRetryRecommendations={() => void refreshModelRecommendations()}
                  installPath={installPath}
                  availableSpace={availableSpace}
                  onChangeModelsDir={handleChangeModelsDir}
                />
              </div>
            )}

            {currentStep === 'installing' && (
              <SetupInstallingStep
                percent={totalProgress}
                runningProgress={runningDownloadProgress}
                installFinished={installFinished}
                downloadError={downloadError}
                onBack={() => {
                  queue.reset()
                  setCurrentStep('location')
                }}
                onRetry={retryInstallation}
              />
            )}
          </div>

          {currentStep !== 'installing' && (
            <div className={styles.footer}>
              {currentStep === 'location' ? (
                <div className={styles.footerCopy}>
                  <SetupLegalDisclaimer className={styles.footerDisclaimer} />
                  <SetupAnalyticsOptOut />
                </div>
              ) : (
                <span />
              )}
              <div className={styles.actions}>
                <Button
                  appearance="brand"
                  hierarchy="primary"
                  size="lg"
                  label={getNextButtonText()}
                  disabled={isNextDisabled()}
                  onClick={() => void handleNext()}
                />
              </div>
            </div>
          )}
          {actionError && (
            <Text as="div" variant="body" size="sm" className={styles.actionError}>
              {actionError}
            </Text>
          )}
        </div>
      </div>
    </TooltipProvider>
  )
}
