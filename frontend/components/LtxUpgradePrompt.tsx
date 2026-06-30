import { useCallback, useEffect, useMemo, useState } from 'react'
import { AlertCircle, ArrowUpRight, Download, Loader2, Sparkles, X } from 'lucide-react'
import { ApiClient, type ApiSuccessOf } from '../lib/api-client'
import { logger } from '../lib/logger'
import { useHfAuth } from '../hooks/use-hf-auth'
import { useHfModelAccess } from '../hooks/use-hf-model-access'
import { useT } from '../lib/i18n'
import { Button } from './ui/button'
import './LtxUpgradePrompt.css'

type UpgradeRecommendation = Extract<ApiSuccessOf<'getLtxRecommendation'>, { status: 'upgrade' }>
const EMPTY_MODEL_TYPES: Parameters<typeof useHfModelAccess>[0] = []

interface LtxUpgradePromptProps {
  recommendation: UpgradeRecommendation
  onClose: () => void
  onComplete: () => Promise<void> | void
}

type UpgradePhase = 'idle' | 'starting' | 'downloading' | 'finishing'

function formatCheckpointId(cpId: string): string {
  return cpId.replace(/-/g, ' ')
}

export function LtxUpgradePrompt({
  recommendation,
  onClose,
  onComplete,
}: LtxUpgradePromptProps) {
  const { t } = useT()
  const [wantsUpgrade, setWantsUpgrade] = useState(false)
  const [phase, setPhase] = useState<UpgradePhase>('idle')
  const [downloadSessionId, setDownloadSessionId] = useState<string | null>(null)
  const [downloadProgress, setDownloadProgress] = useState<ApiSuccessOf<'getModelDownloadProgress'> | null>(
    null,
  )
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const isPreflightPhase = wantsUpgrade && phase === 'idle'
  const authEnabled = isPreflightPhase
  const { hfAuthStatus, hfAuthPolling, startHuggingFaceLogin } = useHfAuth(authEnabled)
  const { accessMap, allAuthorized, checking } = useHfModelAccess(
    authEnabled && hfAuthStatus === 'authenticated' ? recommendation.cps_to_download : EMPTY_MODEL_TYPES,
    hfAuthStatus,
  )

  const canClose = phase === 'idle'
  const needsHfAuth = isPreflightPhase && hfAuthStatus !== 'authenticated'
  const unauthorizedRepos = useMemo(
    () => Object.entries(accessMap).filter(([, status]) => status === 'not_authorized'),
    [accessMap],
  )
  const isAccessPending =
    isPreflightPhase
    && hfAuthStatus === 'authenticated'
    && (checking || (Object.keys(accessMap).length === 0 && !allAuthorized))
  const needsModelAccess = isPreflightPhase && unauthorizedRepos.length > 0
  const canStartUpgrade =
    isPreflightPhase
    && phase === 'idle'
    && hfAuthStatus === 'authenticated'
    && !checking
    && allAuthorized

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
        if (recommendation.cps_to_delete.length > 0) {
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
  }, [downloadSessionId, onComplete, phase, recommendation.cps_to_delete])

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
                  {t('upgrade.title')}
                </div>
                <h2 className="mt-3 text-2xl font-semibold tracking-tight text-white sm:text-[30px]">
                  {t('upgrade.detected')}
                </h2>
                <p className="mt-2 text-sm text-blue-100/78">
                  Upgrade target: <span className="font-medium text-blue-50">{recommendation.ltx_model_id}</span>
                </p>
              </div>
            </div>

            {canClose && (
              <button
                type="button"
                onClick={onClose}
                className="rounded-full p-2 text-blue-100/55 transition-colors hover:bg-white/5 hover:text-white"
                aria-label="Close LTX upgrade prompt"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>

          <div className="mt-6 rounded-[24px] border border-blue-400/16 bg-[linear-gradient(145deg,rgba(10,14,22,0.98),rgba(5,8,14,0.96))] p-5 sm:p-6">
            <p className="text-lg font-semibold leading-snug text-blue-50 sm:text-[22px]">
              {recommendation.upgrade_message ?? 'A better LTX checkpoint is ready for this install.'}
            </p>
            <p className="mt-3 max-w-[44ch] text-sm leading-relaxed text-blue-100/72">
              Your previous checkpoint files will be removed from disk once download completes.
            </p>

            <label className="mt-5 flex cursor-pointer items-center gap-3 rounded-2xl border border-blue-400/12 bg-black/35 px-4 py-3 text-sm text-blue-50/92 transition-colors hover:border-blue-300/22">
              <input
                type="checkbox"
                checked={wantsUpgrade}
                onChange={(event) => setWantsUpgrade(event.target.checked)}
                disabled={!canClose}
                className="h-4 w-4 rounded border-blue-300/40 bg-slate-950 text-blue-500 focus:ring-blue-400"
              />
              <span className="font-medium">{t('upgrade.wantIt')}</span>
            </label>
          </div>

          {wantsUpgrade && (
            <div className="mt-5 space-y-4">
              {needsHfAuth && (
                <div className="rounded-2xl border border-blue-400/12 bg-[#060b14] p-5">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <h3 className="text-sm font-semibold text-white">{t('upgrade.connectHf')}</h3>
                      <p className="mt-1 text-sm text-blue-100/70">
                        Sign in to download this checkpoint upgrade from Hugging Face.
                      </p>
                    </div>
                    <div className="rounded-full border border-blue-400/15 bg-blue-500/10 px-2.5 py-1 text-[11px] font-medium text-blue-200">
                      Required
                    </div>
                  </div>

                  <div className="mt-4">
                    <Button
                      onClick={() => {
                        void startHuggingFaceLogin()
                      }}
                      disabled={hfAuthPolling}
                      className="bg-blue-600 text-white hover:bg-blue-500"
                    >
                      {hfAuthPolling ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Waiting for sign in...
                        </>
                      ) : (
                        'Sign in with Hugging Face'
                      )}
                    </Button>
                  </div>
                </div>
              )}

              {isAccessPending && (
                <div className="rounded-2xl border border-blue-400/12 bg-[#060b14] p-5">
                  <div className="flex items-center gap-3 text-sm text-blue-100/78">
                    <Loader2 className="h-4 w-4 animate-spin text-blue-300" />
                    Checking Hugging Face model access...
                  </div>
                </div>
              )}

              {needsModelAccess && (
                <div className="rounded-2xl border border-blue-400/12 bg-[#060b14] p-5">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <h3 className="text-sm font-semibold text-white">{t('upgrade.acceptAccess')}</h3>
                      <p className="mt-1 text-sm text-blue-100/70">
                        Accept the Hugging Face license for these repos before starting the upgrade.
                      </p>
                    </div>
                    <div className="rounded-full border border-amber-400/18 bg-amber-500/10 px-2.5 py-1 text-[11px] font-medium text-amber-200">
                      Action needed
                    </div>
                  </div>

                  <div className="mt-4 space-y-2.5">
                    {unauthorizedRepos.map(([repoId]) => (
                      <div
                        key={repoId}
                        className="flex items-center justify-between gap-3 rounded-2xl border border-blue-400/10 bg-black/35 px-4 py-3"
                      >
                        <span className="min-w-0 truncate text-xs text-blue-100/75">{repoId}</span>
                        <button
                          type="button"
                          onClick={() => window.electronAPI.openHuggingFaceRepo({ repoId })}
                          className="inline-flex shrink-0 items-center gap-1 text-xs font-medium text-blue-300 transition-colors hover:text-blue-200"
                        >
                          Open repo
                          <ArrowUpRight className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {canStartUpgrade && (
                <div className="ltx-upgrade-reveal rounded-2xl border border-blue-400/12 bg-[#060b14] p-5">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <p className="text-sm font-semibold text-white">{t('upgrade.ready')}</p>
                      <p className="mt-1 text-sm text-blue-100/72">
                        This will download {recommendation.cps_to_download.length} checkpoint
                        {recommendation.cps_to_download.length === 1 ? '' : 's'} and swap out the old primary model.
                      </p>
                    </div>
                    <Button
                      onClick={() => {
                        void handleStartUpgrade()
                      }}
                      className="bg-blue-600 text-white hover:bg-blue-500"
                    >
                      <Download className="mr-2 h-4 w-4" />
                      {t('upgrade.upgradeNow')}
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
                      ? 'Cleaning up the previous checkpoint files.'
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
