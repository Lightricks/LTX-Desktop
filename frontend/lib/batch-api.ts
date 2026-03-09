import type { BatchSubmitRequest, BatchSubmitResponse, BatchStatusResponse } from '@/types/batch'

const getBaseUrl = async (): Promise<string> => {
  if (window.electronAPI) {
    return await window.electronAPI.getBackendUrl()
  }
  return 'http://localhost:8000'
}

export async function submitBatch(request: BatchSubmitRequest): Promise<BatchSubmitResponse> {
  const base = await getBaseUrl()
  const resp = await fetch(`${base}/api/queue/submit-batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!resp.ok) throw new Error(`Batch submit failed: ${resp.status}`)
  return resp.json()
}

export async function getBatchStatus(batchId: string): Promise<BatchStatusResponse> {
  const base = await getBaseUrl()
  const resp = await fetch(`${base}/api/queue/batch/${batchId}/status`)
  if (!resp.ok) throw new Error(`Batch status failed: ${resp.status}`)
  return resp.json()
}

export async function cancelBatch(batchId: string): Promise<void> {
  const base = await getBaseUrl()
  await fetch(`${base}/api/queue/batch/${batchId}/cancel`, { method: 'POST' })
}

export async function retryFailedBatch(batchId: string): Promise<BatchSubmitResponse> {
  const base = await getBaseUrl()
  const resp = await fetch(`${base}/api/queue/batch/${batchId}/retry-failed`, { method: 'POST' })
  if (!resp.ok) throw new Error(`Batch retry failed: ${resp.status}`)
  return resp.json()
}
