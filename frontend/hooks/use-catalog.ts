import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiClient } from '../lib/api-client'
import type { ApiSuccessOf } from '../lib/api-client'
import { catalogVariantKey } from '../lib/lora-library'

export type IcLoraListItem = ApiSuccessOf<'listIcLoras'>['ic_loras'][number]
export type LoraCatalogListItem = ApiSuccessOf<'listLoras'>['loras'][number]

type StartResult = { ok: true; data: { sessionId: string } } | { ok: false; error: { message: string } }
type ProgressResult =
  | { ok: true; data: { progress: number; status: 'downloading' | 'complete' | 'error'; error?: string | null } }
  | { ok: false; error: { message: string } }

// Shared download + poll state machine for a catalog item (LoRA or IC-LoRA). The two hooks
// below differ only in which endpoints they hit and how they list items; this owns the
// identical download/poll/error half. onComplete refreshes the caller's list.
// downloadingKey / downloadError.key are catalogVariantKey(id, variantId) so two checkpoints
// of the same item don't clobber each other's UI state.
function useCatalogDownload(
  startDownload: (id: string, variantId?: string) => Promise<StartResult>,
  getProgress: (sessionId: string) => Promise<ProgressResult>,
  onComplete: () => void,
) {
  const [downloadingKey, setDownloadingKey] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [downloadError, setDownloadError] = useState<{ key: string; message: string } | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const download = useCallback(async (id: string, variantId?: string) => {
    const key = catalogVariantKey(id, variantId)
    setDownloadError(null)
    const start = await startDownload(id, variantId)
    if (!start.ok) {
      setDownloadError({ key, message: start.error.message })
      return
    }
    setDownloadingKey(key)
    setProgress(0)
    const sessionId = start.data.sessionId
    // Never stack intervals: a re-entrant download() would otherwise leak the previous one.
    if (pollRef.current) clearInterval(pollRef.current)
    const stop = () => {
      if (pollRef.current) clearInterval(pollRef.current)
      pollRef.current = null
      setDownloadingKey(null)
      setProgress(0)
    }
    let inFlight = false
    let failures = 0
    pollRef.current = setInterval(async () => {
      if (inFlight) return // don't overlap polls if a tick outruns the 1s interval
      inFlight = true
      try {
        const p = await getProgress(sessionId)
        if (!p.ok) {
          // Tolerate a blip, but give up (and surface it) rather than poll forever.
          if (++failures >= 3) { stop(); setDownloadError({ key, message: 'Lost contact with the download.' }) }
          return
        }
        failures = 0
        setProgress(p.data.progress)
        if (p.data.status !== 'downloading') {
          stop()
          if (p.data.status === 'error') setDownloadError({ key, message: p.data.error ?? 'Download failed' })
          onComplete()
        }
      } finally {
        inFlight = false
      }
    }, 1000)
  }, [startDownload, getProgress, onComplete])

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  return { downloadingKey, progress, downloadError, download }
}

export function useIcLoras(enabled: boolean) {
  const [icLoras, setIcLoras] = useState<IcLoraListItem[]>([])

  const refresh = useCallback(async () => {
    if (!enabled) return
    const r = await ApiClient.listIcLoras()
    if (r.ok) setIcLoras(r.data.ic_loras)
  }, [enabled])

  useEffect(() => { void refresh() }, [refresh])

  const { downloadingKey, progress, downloadError, download: downloadIcLora } = useCatalogDownload(
    // Attach the in-app HF token when the user is signed in; the backend ignores it for public
    // repos and requires it only for gated entries (optional auth).
    (id, variantId) => ApiClient.startIcLoraDownload({
      ic_lora_id: id,
      variant_id: variantId,
      use_hf_auth: true,
    }),
    (sessionId) => ApiClient.getIcLoraDownloadProgress({ sessionId }),
    refresh,
  )

  return { icLoras, refresh, downloadIcLora, downloadingKey, progress, downloadError }
}

// Plain-LoRA catalog: uses /api/loras* (an isolated download session, so it never collides
// with an in-flight IC-LoRA download).
export function useLoraCatalog(enabled: boolean) {
  const [loras, setLoras] = useState<LoraCatalogListItem[]>([])

  const refresh = useCallback(async () => {
    if (!enabled) return
    const r = await ApiClient.listLoras()
    if (r.ok) setLoras(r.data.loras)
  }, [enabled])

  useEffect(() => { void refresh() }, [refresh])

  const { downloadingKey, progress, downloadError, download: downloadLora } = useCatalogDownload(
    (id, variantId) => ApiClient.startLoraDownload({
      lora_id: id,
      variant_id: variantId,
      use_hf_auth: true,
    }),
    (sessionId) => ApiClient.getLoraDownloadProgress({ sessionId }),
    refresh,
  )

  return { loras, refresh, downloadLora, downloadingKey, progress, downloadError }
}
