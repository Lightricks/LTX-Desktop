import { useCallback, useEffect, useState } from 'react'
import { AlertCircle, Download, Loader2, Sparkles, X } from 'lucide-react'
import { ApiClient, type ApiSuccessOf } from '../lib/api-client'
import { logger } from '../lib/logger'
import { Button } from './ui/button'
import './LtxUpgradePrompt.css'

type UpgradeRecommendation = Extract<ApiSuccessOf<'getLtxRecommendation'>, { status: 'upgrade' }>

interface LtxUpgradePromptProps {
  recommendation: UpgradeRecommendation
  onClose: () => void
  // Dismiss permanently for this model id (persisted), so the prompt doesn't return for it.
  onDontShowAgain: () => void
  onComplete: () => Promise<void> | void
}

type UpgradePhase = 'idle' | 'starting' | 'downloading' | 'finishing'

function formatCheckpointId(cpId: string): string {
  return cpId.replace(/-/g, ' ')
}

export function LtxUpgradePrompt({
  recommendation,
  onClose,
  onDontShowAgain,
  onComplete,
}: LtxUpgradePromptProps) {
  const [wantsUpgrade, setWantsUpgrade] = useState(false)
  const [phase, setPhase] = useState<UpgradePhase>('idle')
  const [downloadSessionId, setDownloadSessionId] = useState<string | null>(null)
  const [downloadProgress, setDownloadProgress] = useState<ApiSuccessOf<'getModelDownloadProgress'> | null>(
    null,
  )
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  // Default to reclaiming disk (base models are tens of GB); unchecking keeps the old
  // version installed so it stays selectable in the Settings version manager.
  const [deleteOld, setDeleteOld] = useState(true)

  const hasOldToDelete = recommendation.cps_to_delete.length > 0
  const canClose = phase === 'idle'
  // LTX base-model checkpoints are public on Hugging Face, so the upgrade download needs no
  // sign-in or model-access acceptance — once the user opts in, it can start immediately.
  const canStartUpgrade = wantsUpgrade && phase === 'idle'

  const runningProgress = downloadProgress?.status === 'downloading' ? downloadProgress : null
  const totalProgress = runningProgress?.total_progress ?? (phase === 'finishing' ? 100 : 0)
  const completedCount = runningProgress?.completed_files.length ?? 0
  const totalCount = runningProgress?.all_files.length ?? recommendation.cps_to_download.length

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
    if (phase !== 'downloading' || !downloadSessionId) return

    let cancelled = false
    const pollProgress = async () => {
      const progressResult = await ApiClient.getModelDownloadProgress({ sessionId: downloadSessionId })
      if (!progressResult.ok) {
        logger.warn(`Failed polling LTX upgrade progress: ${progressResult.error.message}`)
        return
      }
      if (cancelled) return

      const progress = progressResult.data
      setDownloadProgress(progress)

      if (progress.status === 'error') {
        setPhase('idle')
        setErrorMessage(progress.error || 'Upgrade download failed.')
        return
      }

      if (progress.status === 'complete') {
        setPhase('finishing')
        if (deleteOld && recommendation.cps_to_delete.length > 0) {
          const deleteResult = await ApiClient.deleteModels({ cp_ids: recommendation.cps_to_delete })
          if (!deleteResult.ok) {
            logger.error(`Failed finalizing LTX upgrade: ${deleteResult.error.message}`)
            if (cancelled) return
            setPhase('idle')
            setErrorMessage(deleteResult.error.message)
            return
          }
        }
        try {
          await onComplete()
          if (cancelled) return
          // Reset to idle rather than waiting for the parent's refresh to flip away from 'upgrade'
          // — if the active model isn't flipped yet we'd otherwise be stuck on "Finishing up...".
          setPhase('idle')
          onClose()
        } catch (e) {
          logger.error(`Failed finalizing LTX upgrade: ${e}`)
          if (cancelled) return
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
  }, [deleteOld, downloadSessionId, onClose, onComplete, phase, recommendation.cps_to_delete])

  const handleStartUpgrade = useCallback(async () => {
    if (!canStartUpgrade) return

    setErrorMessage(null)
    setDownloadProgress(null)
    setPhase('starting')

    const result = await ApiClient.startModelDownload({
      type: 'upgrade',
      cp_ids: recommendation.cps_to_download,
    })
    if (!result.ok) {
      logger.warn(`Failed to start LTX upgrade download: ${result.error.message}`)
      setPhase('idle')
      setErrorMessage(result.error.message)
      return
    }

    const response = result.data
    if (response.status !== 'started') {
      setPhase('idle')
      setErrorMessage('Unexpected response while starting the upgrade.')
      return
    }
    setDownloadSessionId(response.sessionId)
    setPhase('downloading')
  }, [canStartUpgrade, recommendation.cps_to_download])

  return (
    <div className="ltx-upgrade-backdrop fixed inset-0 z-[55] flex items-center justify-center bg-black/72 px-4 py-6 backdrop-blur-sm">
      <div
        className="absolute inset-0"
        onClick={() => {
          if (canClose) onClose()
        }}
      />

      <div className="ltx-upgrade-card relative w-full max-w-[640px] overflow-hidden rounded-[28px] border border-blue-500/20 bg-[#04070d] shadow-[0_24px_120px_rgba(0,0,0,0.72)]">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(59,130,246,0.16),_transparent_44%),linear-gradient(180deg,rgba(15,23,42,0.68),rgba(2,6,23,0.16))]" />
        <div className="relative px-6 pb-6 pt-6 sm:px-8">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start gap-4">
              <div className="mt-0.5 flex h-12 w-12 items-center justify-center rounded-2xl border border-blue-400/20 bg-blue-500/12">
                <Sparkles className="h-5 w-5 text-blue-200" />
              </div>
              <div>
                <div className="inline-flex items-center rounded-full border border-blue-400/20 bg-blue-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-blue-200/85">
                  Optional Upgrade
                </div>
                <h2 className="mt-3 text-2xl font-semibold tracking-tight text-white sm:text-[30px]">
                  LTX Model Upgrade Detected!
                </h2>
                <p className="mt-2 text-sm text-blue-100/78">
                  Upgrade target: <span className="font-medium text-blue-50">{recommendation.ltx_model_id}</span>
                </p>
              </div>
            </div>

            {canClose && (
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={onDontShowAgain}
                  className="rounded-full px-3 py-1.5 text-xs text-blue-100/55 transition-colors hover:bg-white/5 hover:text-white"
                >
                  Don't show again
                </button>
                <button
                  type="button"
                  onClick={onClose}
                  className="rounded-full p-2 text-blue-100/55 transition-colors hover:bg-white/5 hover:text-white"
                  aria-label="Close LTX upgrade prompt"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            )}
          </div>

          <div className="mt-6 rounded-[24px] border border-blue-400/16 bg-[linear-gradient(145deg,rgba(10,14,22,0.98),rgba(5,8,14,0.96))] p-5 sm:p-6">
            {recommendation.upgrade_message ? (
              <>
                <p className="text-lg font-semibold leading-snug text-blue-50 sm:text-[22px]">What's new in this version</p>
                <ul className="mt-3 space-y-1.5">
                  {recommendation.upgrade_message.split('\n').map((line) => line.trim()).filter(Boolean).map((line, i) => (
                    <li key={i} className="flex gap-2 text-sm leading-relaxed text-blue-100/82">
                      <span className="mt-0.5 text-blue-300/70">•</span>
                      <span>{line}</span>
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p className="text-lg font-semibold leading-snug text-blue-50 sm:text-[22px]">
                A better LTX checkpoint is ready for this install.
              </p>
            )}
            {hasOldToDelete && (
              <p className="mt-3 max-w-[44ch] text-sm leading-relaxed text-blue-100/72">
                {deleteOld
                  ? 'Your previous checkpoint will be removed from disk once the download completes.'
                  : 'Your previous checkpoint will be kept — switch between versions anytime in Settings → Models.'}
              </p>
            )}

            <label className="mt-5 flex cursor-pointer items-center gap-3 rounded-2xl border border-blue-400/12 bg-black/35 px-4 py-3 text-sm text-blue-50/92 transition-colors hover:border-blue-300/22">
              <input
                type="checkbox"
                checked={wantsUpgrade}
                onChange={(event) => setWantsUpgrade(event.target.checked)}
                disabled={!canClose}
                className="h-4 w-4 rounded border-blue-300/40 bg-slate-950 text-blue-500 focus:ring-blue-400"
              />
              <span className="font-medium">I want this!</span>
            </label>

            {wantsUpgrade && hasOldToDelete && (
              <label className="mt-3 flex cursor-pointer items-center gap-3 rounded-2xl border border-blue-400/12 bg-black/35 px-4 py-3 text-sm text-blue-50/92 transition-colors hover:border-blue-300/22">
                <input
                  type="checkbox"
                  checked={deleteOld}
                  onChange={(event) => setDeleteOld(event.target.checked)}
                  disabled={!canClose}
                  className="h-4 w-4 rounded border-blue-300/40 bg-slate-950 text-blue-500 focus:ring-blue-400"
                />
                <span className="font-medium">Delete the previous checkpoint to free up disk space</span>
              </label>
            )}
          </div>

          {wantsUpgrade && (
            <div className="mt-5 space-y-4">
              {canStartUpgrade && (
                <div className="ltx-upgrade-reveal rounded-2xl border border-blue-400/12 bg-[#060b14] p-5">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <p className="text-sm font-semibold text-white">Ready when you are</p>
                      <p className="mt-1 text-sm text-blue-100/72">
                        This will download {recommendation.cps_to_download.length} checkpoint
                        {recommendation.cps_to_download.length === 1 ? '' : 's'}
                        {hasOldToDelete && deleteOld ? ' and remove the old one.' : '.'}
                      </p>
                    </div>
                    <Button
                      onClick={() => {
                        void handleStartUpgrade()
                      }}
                      className="bg-blue-600 text-white hover:bg-blue-500"
                    >
                      <Download className="mr-2 h-4 w-4" />
                      Upgrade now
                    </Button>
                  </div>
                </div>
              )}
            </div>
          )}

          {phase !== 'idle' && (
            <div className="mt-5 rounded-2xl border border-blue-400/12 bg-[#060b14] p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h3 className="text-sm font-semibold text-white">
                    {phase === 'starting' ? 'Preparing your upgrade...' : phase === 'downloading' ? 'Downloading update...' : 'Finishing up...'}
                  </h3>
                  <p className="mt-1 text-sm text-blue-100/70">
                    {phase === 'finishing'
                      ? (deleteOld && hasOldToDelete ? 'Cleaning up the previous checkpoint files.' : 'Finalizing the upgrade.')
                      : 'Keep this window open while the new checkpoint is downloaded.'}
                  </p>
                </div>
                <div className="inline-flex items-center gap-2 rounded-full border border-blue-400/14 bg-blue-500/10 px-3 py-1 text-xs text-blue-100/82">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Working
                </div>
              </div>

              <div className="mt-5">
                <div className="mb-2 flex items-center justify-between text-xs text-blue-100/72">
                  <span>
                    {phase === 'starting'
                      ? 'Waiting for the download session to start'
                      : `Checkpoint progress ${Math.min(completedCount + (runningProgress ? 1 : 0), totalCount)} / ${totalCount}`}
                  </span>
                  <span>{Math.round(totalProgress)}%</span>
                </div>

                <div className="h-2 overflow-hidden rounded-full bg-slate-950/60">
                  {phase === 'starting' ? (
                    <div className="ltx-upgrade-indeterminate h-full bg-blue-500/55" />
                  ) : (
                    <div
                      className="h-full rounded-full bg-[linear-gradient(90deg,#60a5fa,#2563eb)] transition-all duration-300"
                      style={{ width: `${Math.max(totalProgress, 4)}%` }}
                    />
                  )}
                </div>

                {runningProgress?.current_downloading_file && (
                  <div className="mt-3 space-y-1">
                    <div className="flex items-center justify-between text-xs text-blue-100/70">
                      <span className="truncate">Current file</span>
                      <span>{Math.round(runningProgress.current_file_progress)}%</span>
                    </div>
                    <div className="truncate text-sm text-blue-50/85">
                      {formatCheckpointId(runningProgress.current_downloading_file)}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {errorMessage && (
            <div className="mt-5 flex items-start gap-3 rounded-2xl border border-red-500/25 bg-red-500/10 px-4 py-3 text-sm text-red-100">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-300" />
              <span>{errorMessage}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
