import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiClient, type ApiRequestBodyOf, type ApiSuccessOf } from '@/lib/api-client'
import {
  createDownloadProgressPollGuard,
  mapLoraProgressToDownloadProgress,
  resolveDownloadFailure,
  type FirstRunDownloadStep,
  type FirstRunLoraItem,
} from '@/lib/first-run-downloads'
import { logger } from '@/lib/logger'

type StartModelDownloadBody = NonNullable<ApiRequestBodyOf<'startModelDownload'>>
export type DownloadQueueCheckpointID = NonNullable<StartModelDownloadBody['cp_ids']>[number]
type DownloadQueueProgress = ApiSuccessOf<'getModelDownloadProgress'>

export type DownloadQueueItem =
  | { kind: 'model-step'; step: FirstRunDownloadStep<DownloadQueueCheckpointID> }
  | { kind: 'lora'; item: FirstRunLoraItem }

interface UseDownloadQueueOptions {
  enabled: boolean
  onComplete: () => void
}

interface UseDownloadQueueResult {
  progress: DownloadQueueProgress | null
  error: string | null
  start: (items: readonly DownloadQueueItem[]) => Promise<void>
  fail: (message: string) => void
  reset: () => void
}

const DOWNLOAD_PROGRESS_POLL_MS = 500

// Owns the first-run install queue: model steps then LoRAs, drained FIFO with
// one session, one poll effect, and one failure branch. Blocking model failures
// surface as `error`; non-blocking model failures skip ahead; LoRA failures
// (start, progress, or poll) log and continue without blocking setup.
export function useDownloadQueue({ enabled, onComplete }: UseDownloadQueueOptions): UseDownloadQueueResult {
  const [progress, setProgress] = useState<DownloadQueueProgress | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const queueRef = useRef<DownloadQueueItem[]>([])
  const currentRef = useRef<DownloadQueueItem | null>(null)
  const onCompleteRef = useRef(onComplete)
  onCompleteRef.current = onComplete

  const finish = useCallback(() => {
    queueRef.current = []
    currentRef.current = null
    setSessionId(null)
    onCompleteRef.current()
  }, [])

  const fail = useCallback((message: string) => {
    queueRef.current = []
    currentRef.current = null
    setSessionId(null)
    setError(message)
  }, [])

  const reset = useCallback(() => {
    queueRef.current = []
    currentRef.current = null
    setSessionId(null)
    setProgress(null)
    setError(null)
  }, [])

  const advance = useCallback(async (): Promise<void> => {
    const next = queueRef.current.shift() ?? null
    currentRef.current = next
    if (!next) {
      finish()
      return
    }
    if (next.kind === 'lora') {
      setSessionId(null)
      setError(null)
      const result =
        next.item.kind === 'ic-lora'
          ? await ApiClient.startIcLoraDownload({ ic_lora_id: next.item.id, use_hf_auth: true })
          : await ApiClient.startLoraDownload({ lora_id: next.item.id, use_hf_auth: true })
      if (!result.ok) {
        logger.error(`LoRA download failed to start; continuing setup: ${result.error.message}`)
        currentRef.current = null
        await advance()
        return
      }
      setProgress({
        status: 'downloading',
        all_files: [next.item.name],
        completed_files: [],
        current_downloading_file: next.item.name,
        current_file_progress: 0,
        expected_total_bytes: next.item.sizeBytes,
        speed_bytes_per_sec: 0,
        total_downloaded_bytes: 0,
        total_progress: 0,
        error: null,
      } as DownloadQueueProgress)
      setSessionId(result.data.sessionId)
      return
    }
    setProgress(null)
    setError(null)
    try {
      const result = await ApiClient.startModelDownload({
        type: next.step.type,
        cp_ids: next.step.cpIds,
      })
      if (!result.ok) {
        throw new Error(result.error.message)
      }
      if (result.data.status !== 'started') {
        throw new Error('Unexpected response while starting model download.')
      }
      setSessionId(result.data.sessionId)
    } catch (e) {
      const resolution = resolveDownloadFailure(next.step, queueRef.current)
      if (resolution.action === 'error') {
        queueRef.current = []
        currentRef.current = null
        logger.error(`Download start error: ${e}`)
        setError(e instanceof Error ? e.message : 'Failed to start model download.')
        return
      }
      logger.error(
        `Recommended quality download failed to start; continuing setup: ${
          e instanceof Error ? e.message : String(e)
        }`,
      )
      if (resolution.action === 'continue') {
        // resolution.next is queueRef[0]; advance() shifts and starts it.
        await advance()
        return
      }
      setSessionId(null)
      await advance()
    }
  }, [finish])

  const start = useCallback(
    async (items: readonly DownloadQueueItem[]): Promise<void> => {
      queueRef.current = [...items]
      currentRef.current = null
      setError(null)
      setProgress(null)
      setSessionId(null)
      await advance()
    },
    [advance],
  )

  useEffect(() => {
    if (!enabled || !sessionId) return
    const item = currentRef.current
    if (!item) return

    const pollGuard = createDownloadProgressPollGuard()
    let cancelled = false
    const pollProgress = async () => {
      if (!pollGuard.begin(sessionId)) return
      try {
        if (item.kind === 'lora') {
          const lora = item.item
          const result =
            lora.kind === 'ic-lora'
              ? await ApiClient.getIcLoraDownloadProgress({ sessionId })
              : await ApiClient.getLoraDownloadProgress({ sessionId })
          if (cancelled) return
          if (!result.ok) {
            logger.error(`LoRA progress poll error: ${result.error.message}`)
            return
          }
          const loraProgress = result.data
          if (loraProgress.status === 'error' || loraProgress.status === 'complete') {
            pollGuard.markTerminal(sessionId)
          }
          setProgress(mapLoraProgressToDownloadProgress(loraProgress, lora) as DownloadQueueProgress)
          if (loraProgress.status === 'error') {
            logger.error(`LoRA download failed; continuing setup: ${loraProgress.error || 'Download failed.'}`)
            setSessionId(null)
            await advance()
          } else if (loraProgress.status === 'complete') {
            setSessionId(null)
            await advance()
          }
          return
        }
        const result = await ApiClient.getModelDownloadProgress({ sessionId })
        if (cancelled) return
        if (!result.ok) {
          logger.error(`Progress poll error: ${result.error.message}`)
          return
        }
        const modelProgress = result.data
        if (modelProgress.status === 'error' || modelProgress.status === 'complete') {
          pollGuard.markTerminal(sessionId)
        }
        setProgress(modelProgress)
        if (modelProgress.status === 'error') {
          const resolution = resolveDownloadFailure(item.step, queueRef.current)
          if (resolution.action === 'error') {
            queueRef.current = []
            currentRef.current = null
            setError(modelProgress.error || 'Download failed.')
            return
          }
          logger.error(
            `Recommended quality download failed; continuing setup: ${modelProgress.error || 'Download failed.'}`,
          )
          if (resolution.action === 'continue') {
            await advance()
            return
          }
          setSessionId(null)
          await advance()
        } else if (modelProgress.status === 'complete') {
          setSessionId(null)
          await advance()
        }
      } finally {
        pollGuard.end()
      }
    }

    void pollProgress()
    const interval = setInterval(() => void pollProgress(), DOWNLOAD_PROGRESS_POLL_MS)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [enabled, sessionId, advance])

  return { progress, error, start, fail, reset }
}
