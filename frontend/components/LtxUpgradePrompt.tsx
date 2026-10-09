import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AlertCircle, X } from 'lucide-react'
import { useHfAuth } from '../hooks/use-hf-auth'
import { useHfModelAccess } from '../hooks/use-hf-model-access'
import { ApiClient, type ApiRequestBodyOf, type ApiSuccessOf } from '../lib/api-client'
import { logger } from '../lib/logger'
import {
  activeDownloadMatchesUpgrade,
  clearUpgradeDeleteOld,
  hasStoredUpgradeDeleteOld,
  readUpgradeDeleteOld,
  upgradePromptVisible,
  writeUpgradeDeleteOld,
} from '../lib/upgrade-prompt-download'
import { HfModelAccessGate } from './HfModelAccessGate'
import { Button } from '@ds/Button/Button'
import {
  AppProductDialogShell,
  ProductDialogBlock,
  ProductDialogCheckboxRow,
  ProductDialogIconButton,
} from './dialog/AppProductDialogShell'

type UpgradeRecommendation = Extract<ApiSuccessOf<'getLtxRecommendation'>, { status: 'upgrade' }>
type ModelCheckpointID = NonNullable<
  NonNullable<ApiRequestBodyOf<'checkModelAccess'>>['cp_ids']
>[number]

type UpgradePhase = 'idle' | 'starting' | 'downloading' | 'finishing'

/** Frozen state for local UI previews. Production leaves this unset. */
export type LtxUpgradePromptPreview = {
  phase?: UpgradePhase
  downloadProgress?: ApiSuccessOf<'getModelDownloadProgress'> | null
  deleteOld?: boolean
  assumeAuthorized?: boolean
  frozen?: boolean
}

interface LtxUpgradePromptProps {
  recommendation: UpgradeRecommendation
  onClose: () => void
  onComplete: () => Promise<void> | void
  preview?: LtxUpgradePromptPreview
}

export function LtxUpgradePrompt({
  recommendation,
  onClose,
  onComplete,
  preview,
}: LtxUpgradePromptProps) {
  // Vite replaces this with `false` in a production build, so the preview branches drop out.
  const devPreview = import.meta.env.DEV ? preview : undefined
  const frozen = devPreview?.frozen === true
  const defaultDeleteOld = devPreview?.deleteOld ?? !recommendation.loses_built_in_control
  const [phase, setPhase] = useState<UpgradePhase>(devPreview?.phase ?? 'idle')
  const [checkingActiveDownload, setCheckingActiveDownload] = useState(
    !frozen && (devPreview?.phase ?? 'idle') === 'idle',
  )
  // True only when this mount is a reload of a download the modal already started.
  const resumedDownloadRef = useRef(!frozen && hasStoredUpgradeDeleteOld(recommendation.ltx_model_id))
  const [background, setBackground] = useState(resumedDownloadRef.current)
  const [downloadSessionId, setDownloadSessionId] = useState<string | null>(null)
  const [downloadProgress, setDownloadProgress] = useState<ApiSuccessOf<'getModelDownloadProgress'> | null>(
    devPreview?.downloadProgress ?? null,
  )
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  // Default off when deleting the old bundle would also wipe built-in Union Control IC-LoRA
  // (2.3 → 2.5). Otherwise default on to reclaim the tens of GB the old transformer uses.
  const [deleteOld, setDeleteOld] = useState(() =>
    frozen ? defaultDeleteOld : readUpgradeDeleteOld(recommendation.ltx_model_id, defaultDeleteOld),
  )

  const cpsToDownload = useMemo(
    () => (frozen ? [] : recommendation.cps_to_download) as ModelCheckpointID[],
    [frozen, recommendation.cps_to_download],
  )
  const { hfAuthStatus, hfAuthPolling, startHuggingFaceLogin } = useHfAuth(!frozen)
  const { accessMap, allAuthorized, checking: checkingAccess, checkError, recheckAccess } = useHfModelAccess(
    cpsToDownload,
    hfAuthStatus,
  )

  const hasOldToDelete = recommendation.cps_to_delete.length > 0
  const canClose = phase === 'idle'
  const modelAccessReady = devPreview?.assumeAuthorized === true || (allAuthorized && !checkingAccess)
  const canStartUpgrade = phase === 'idle' && modelAccessReady && !frozen && !checkingActiveDownload

  const runningProgress = downloadProgress?.status === 'downloading' ? downloadProgress : null
  const totalProgress = runningProgress?.total_progress ?? (phase === 'finishing' ? 100 : 0)

  useEffect(() => {
    if (!canClose) return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [canClose, onClose])

  useEffect(() => {
    if (frozen) return
    let cancelled = false
    const reattach = async () => {
      const active = await ApiClient.getActiveDownload()
      if (cancelled) return
      if (!active.ok) {
        logger.warn(`Failed to read the active model download: ${active.error.message}`)
      }
      const startedHere = hasStoredUpgradeDeleteOld(recommendation.ltx_model_id)
      if (
        startedHere
        && active.ok
        && active.data.session_id
        && activeDownloadMatchesUpgrade(active.data.cp_ids ?? [], recommendation.cps_to_download)
      ) {
        setDownloadSessionId(active.data.session_id)
        setPhase('downloading')
        if (resumedDownloadRef.current) setBackground(true)
      } else if (resumedDownloadRef.current) {
        setBackground(false)
      }
      setCheckingActiveDownload(false)
    }
    void reattach()
    return () => {
      cancelled = true
    }
  }, [frozen, recommendation.cps_to_download, recommendation.ltx_model_id])

  useEffect(() => {
    if (frozen) return
    if (phase !== 'downloading' || !downloadSessionId) return

    let cancelled = false
    let finalizing = false
    const pollProgress = async () => {
      if (cancelled || finalizing) return
      const progressResult = await ApiClient.getModelDownloadProgress({ sessionId: downloadSessionId })
      if (!progressResult.ok) {
        logger.warn(`Failed polling LTX upgrade progress: ${progressResult.error.message}`)
        return
      }
      if (cancelled || finalizing) return

      const progress = progressResult.data
      setDownloadProgress(progress)

      if (progress.status === 'error') {
        setPhase('idle')
        setErrorMessage(progress.error || 'Upgrade download failed.')
        return
      }

      if (progress.status === 'complete') {
        // Overlapping polls both observe `complete`. Claim finalize before the
        // next await so the second poll cannot delete the old checkpoint twice.
        finalizing = true
        setPhase('finishing')
        if (deleteOld && recommendation.cps_to_delete.length > 0) {
          const deleteResult = await ApiClient.deleteModels({ cp_ids: recommendation.cps_to_delete })
          if (!deleteResult.ok) {
            logger.error(`Failed finalizing LTX upgrade: ${deleteResult.error.message}`)
            // `setPhase('finishing')` re-runs this effect and marks the in-flight
            // poll cancelled. Still return to idle so the prompt is not stuck.
            finalizing = false
            setPhase('idle')
            setErrorMessage(deleteResult.error.message)
            return
          }
        }
        clearUpgradeDeleteOld(recommendation.ltx_model_id)
        try {
          await onComplete()
          if (cancelled) return
          // Reset to idle rather than waiting for the parent's refresh to flip away from 'upgrade'
          // — if the active model isn't flipped yet we'd otherwise be stuck on "Finishing up...".
          // Do not call onClose: that persists a dismissal, and a later removal of this
          // checkpoint should be allowed to show the prompt again.
          setPhase('idle')
        } catch (e) {
          logger.error(`Failed finalizing LTX upgrade: ${e}`)
          finalizing = false
          setPhase('idle')
          setErrorMessage(e instanceof Error ? e.message : 'Upgrade downloaded, but cleanup failed.')
        }
      }
    }

    void pollProgress()
    const interval = setInterval(() => {
      void pollProgress()
    }, 700)

    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [deleteOld, downloadSessionId, frozen, onComplete, phase, recommendation.cps_to_delete, recommendation.ltx_model_id])

  const handleStartUpgrade = useCallback(async () => {
    if (!canStartUpgrade || frozen) return

    setErrorMessage(null)
    setDownloadProgress(null)
    writeUpgradeDeleteOld(recommendation.ltx_model_id, deleteOld)
    setPhase('starting')

    const result = await ApiClient.startModelDownload({
      type: 'upgrade',
      cp_ids: recommendation.cps_to_download,
    })
    if (!result.ok) {
      logger.warn(`Failed to start LTX upgrade download: ${result.error.message}`)
      clearUpgradeDeleteOld(recommendation.ltx_model_id)
      setPhase('idle')
      setErrorMessage(result.error.message)
      return
    }

    const response = result.data
    if (response.status !== 'started') {
      clearUpgradeDeleteOld(recommendation.ltx_model_id)
      setPhase('idle')
      setErrorMessage('Unexpected response while starting the upgrade.')
      return
    }
    setDownloadSessionId(response.sessionId)
    setPhase('downloading')
  }, [canStartUpgrade, deleteOld, frozen, recommendation.cps_to_download, recommendation.ltx_model_id])

  const upgradeReady = modelAccessReady

  const loadingPercentLabel = `${Math.round(phase === 'starting' ? 0 : totalProgress)}%`

  const handleContinueGenerating = () => {
    if (frozen) {
      onClose()
      return
    }
    setBackground(true)
  }

  if (!upgradePromptVisible({ frozen, background })) {
    return null
  }

  return (
    <AppProductDialogShell
      aria-labelledby="ltx-upgrade-dialog-title"
      title="New LTX model available"
      subtitle={recommendation.ltx_model_id}
      onBackdropClick={canClose ? onClose : undefined}
      headerActions={
        canClose ? (
          <ProductDialogIconButton type="button" onClick={onClose} aria-label="Close LTX upgrade prompt">
            <X className="h-4 w-4" />
          </ProductDialogIconButton>
        ) : undefined
      }
      footer={
        phase === 'idle' ? (
          <Button
            appearance="brand"
            hierarchy="primary"
            size="xl"
            className="w-fit"
            label="Update model"
            disabled={!upgradeReady || checkingActiveDownload}
            onClick={() => {
              void handleStartUpgrade()
            }}
          />
        ) : (
          <Button
            appearance="brand"
            hierarchy="primary"
            size="xl"
            className="w-fit"
            label={`Generate while this downloads · ${loadingPercentLabel}`}
            onClick={handleContinueGenerating}
          />
        )
      }
    >
      {recommendation.upgrade_message ? (
        <ProductDialogBlock title="What's new" tone="highlight">
          <ul className="list-disc space-y-1 pl-4 text-sm leading-relaxed text-fg-secondary">
            {recommendation.upgrade_message
              .split('\n')
              .map((line) => line.trim())
              .filter(Boolean)
              .map((line, i) => (
                <li key={i}>{line}</li>
              ))}
          </ul>
        </ProductDialogBlock>
      ) : phase === 'idle' ? (
        <p className="text-sm text-fg-secondary">A better LTX checkpoint is ready for this install.</p>
      ) : null}

      {phase === 'idle' ? (
        <>
          {hasOldToDelete ? (
            <div>
              <ProductDialogCheckboxRow checked={deleteOld} onChange={setDeleteOld} disabled={!canClose}>
                {recommendation.loses_built_in_control
                  ? 'Remove previous checkpoint and built-in control models'
                  : 'Remove previous checkpoint to free disk space'}
              </ProductDialogCheckboxRow>
            </div>
          ) : null}

          {devPreview?.assumeAuthorized !== true && !allAuthorized ? (
            <HfModelAccessGate
              accessMap={accessMap}
              allAuthorized={allAuthorized}
              hfAuthStatus={hfAuthStatus}
              hfAuthPolling={hfAuthPolling}
              startHuggingFaceLogin={() => {
                void startHuggingFaceLogin()
              }}
              checkError={checkError}
              onRetryCheck={recheckAccess}
            />
          ) : null}
        </>
      ) : null}

      {errorMessage ? (
        <p className="flex items-start gap-2 text-sm text-red-600 dark:text-red-300">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{errorMessage}</span>
        </p>
      ) : null}
    </AppProductDialogShell>
  )
}
