import { useState, useRef, useCallback, useEffect } from 'react'
import type { BatchSubmitRequest, BatchStatusResponse, BatchReport } from '@/types/batch'
import { submitBatch, getBatchStatus, cancelBatch, retryFailedBatch } from '@/lib/batch-api'

export interface UseBatchReturn {
  activeBatchId: string | null
  batchStatus: BatchStatusResponse | null
  batchReport: BatchReport | null
  isRunning: boolean
  submit: (request: BatchSubmitRequest) => Promise<void>
  cancel: () => Promise<void>
  retryFailed: () => Promise<void>
  reset: () => void
}

export function useBatch(): UseBatchReturn {
  const [activeBatchId, setActiveBatchId] = useState<string | null>(null)
  const [batchStatus, setBatchStatus] = useState<BatchStatusResponse | null>(null)
  const [batchReport, setBatchReport] = useState<BatchReport | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const startPolling = useCallback((batchId: string) => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const status = await getBatchStatus(batchId)
        setBatchStatus(status)
        if (status.report) {
          setBatchReport(status.report)
          stopPolling()
          playCompletionSound()
        }
      } catch {
        // Ignore polling errors
      }
    }, 1000)
  }, [stopPolling])

  const submit = useCallback(async (request: BatchSubmitRequest) => {
    const response = await submitBatch(request)
    setActiveBatchId(response.batch_id)
    setBatchReport(null)
    startPolling(response.batch_id)
  }, [startPolling])

  const cancel = useCallback(async () => {
    if (activeBatchId) {
      await cancelBatch(activeBatchId)
    }
  }, [activeBatchId])

  const retryFailed = useCallback(async () => {
    if (activeBatchId) {
      await retryFailedBatch(activeBatchId)
      startPolling(activeBatchId)
    }
  }, [activeBatchId, startPolling])

  const reset = useCallback(() => {
    stopPolling()
    setActiveBatchId(null)
    setBatchStatus(null)
    setBatchReport(null)
  }, [stopPolling])

  useEffect(() => stopPolling, [stopPolling])

  const isRunning = batchStatus !== null && batchStatus.report === null

  return { activeBatchId, batchStatus, batchReport, isRunning, submit, cancel, retryFailed, reset }
}

function playCompletionSound(): void {
  try {
    const audio = new Audio('/sounds/batch-complete.mp3')
    audio.volume = 0.5
    audio.play().catch(() => {})
  } catch {
    // Sound not critical
  }
}
