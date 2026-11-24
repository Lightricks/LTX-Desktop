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
  getActiveSession?: () => Promise<{ sessionId: string; key: string; progress: number } | null>,
) {
  const [downloadingKey, setDownloadingKey] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [downloadError, setDownloadError] = useState<{ key: string; message: string } | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pollGenRef = useRef(0)
  const dismissedRef = useRef(false)

  const stopPolling = useCallback(() => {
    pollGenRef.current += 1
    if (pollRef.current) clearInterval(pollRef.current)
    pollRef.current = null
    setDownloadingKey(null)
    setProgress(0)
  }, [])

  const startPolling = useCallback((sessionId: string, key: string, initialProgress = 0) => {
    // Never stack intervals: a re-entrant start/resume would otherwise leak the previous one.
    if (pollRef.current) clearInterval(pollRef.current)
    const pollGen = ++pollGenRef.current
    setDownloadingKey(key)
    setProgress(initialProgress)
    let inFlight = false
    let failures = 0
    const tick = async () => {
      if (pollGen !== pollGenRef.current) return
      if (inFlight) return // don't overlap polls if a tick outruns the 1s interval
      inFlight = true
      try {
        const p = await getProgress(sessionId)
        if (pollGen !== pollGenRef.current) return
        if (!p.ok) {
          // Tolerate a blip, but give up (and surface it) rather than poll forever.
          if (++failures >= 3) { stopPolling(); setDownloadError({ key, message: 'Lost contact with the download.' }) }
          return
        }
        failures = 0
        setProgress(p.data.progress)
        if (p.data.status !== 'downloading') {
          stopPolling()
          if (p.data.status === 'error') setDownloadError({ key, message: p.data.error ?? 'Download failed' })
          onComplete()
        }
      } finally {
        inFlight = false
      }
    }
    void tick()
    pollRef.current = setInterval(() => { void tick() }, 1000)
  }, [getProgress, onComplete, stopPolling])

  const download = useCallback(async (id: string, variantId?: string) => {
    const key = catalogVariantKey(id, variantId)
    dismissedRef.current = false
    setDownloadError(null)
    const start = await startDownload(id, variantId)
    if (dismissedRef.current) return
    if (!start.ok) {
      const active = await getActiveSession?.()
      if (dismissedRef.current) return
      if (active && active.key === key) {
        startPolling(active.sessionId, key, active.progress)
        return
      }
      setDownloadError({ key, message: start.error.message })
      return
    }
    startPolling(start.data.sessionId, key)
  }, [getActiveSession, startDownload, startPolling])

  // Reattach to a download that was already running when this hook mounted (e.g. a recipe
  // screen reopened mid-download). No-op if we're already polling a session.
  const resume = useCallback((sessionId: string, key: string, initialProgress = 0) => {
    if (dismissedRef.current) return
    if (pollRef.current) return
    startPolling(sessionId, key, initialProgress)
  }, [startPolling])

  const cancel = useCallback(() => {
    dismissedRef.current = true
    stopPolling()
  }, [stopPolling])

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current) }, [])

  return { downloadingKey, progress, downloadError, download, resume, cancel }
}

export type CatalogStatus = 'loading' | 'error' | 'loaded'

type IcLoraCatalogApi = Pick<
  typeof ApiClient,
  'listIcLoras' | 'startIcLoraDownload' | 'getIcLoraDownloadProgress' | 'getIcLoraDownloadActive'
>

export function useIcLoras(
  enabled: boolean,
  api: IcLoraCatalogApi = ApiClient,
  options?: { trackDownloads?: boolean },
) {
  const [icLoras, setIcLoras] = useState<IcLoraListItem[]>([])
  const [catalogStatus, setCatalogStatus] = useState<CatalogStatus>('loading')

  const refresh = useCallback(async (): Promise<IcLoraListItem[] | null> => {
    if (!enabled) return null
    setCatalogStatus((status) => (status === 'loaded' ? status : 'loading'))
    const r = await api.listIcLoras()
    if (r.ok) {
      setIcLoras(r.data.ic_loras)
      setCatalogStatus('loaded')
      return r.data.ic_loras
    }
    setCatalogStatus('error')
    return null
  }, [api, enabled])

  useEffect(() => { void refresh() }, [refresh])

  const trackDownloads = options?.trackDownloads ?? true

  const getActiveSession = useCallback(async () => {
    if (!trackDownloads) return null
    const active = await api.getIcLoraDownloadActive()
    if (!active.ok || !active.data.session_id || !active.data.ic_lora_id) return null
    return {
      sessionId: active.data.session_id,
      key: catalogVariantKey(active.data.ic_lora_id),
      progress: active.data.progress ?? 0,
    }
  }, [api, trackDownloads])

  const { downloadingKey, progress, downloadError, download: downloadIcLora, resume, cancel } = useCatalogDownload(
    // Attach the in-app HF token when the user is signed in; the backend ignores it for public
    // repos and requires it only for gated entries (optional auth).
    (id, variantId) => api.startIcLoraDownload({
      ic_lora_id: id,
      variant_id: variantId,
      use_hf_auth: true,
    }),
    (sessionId) => api.getIcLoraDownloadProgress({ sessionId }),
    refresh,
    trackDownloads ? getActiveSession : undefined,
  )

  // Same reattachment the plain-LoRA catalog does: an IC-LoRA download survives this hook
  // unmounting (tab switch) and the renderer reloading, so pick it back up instead of
  // showing an idle Download button that answers DOWNLOAD_ALREADY_RUNNING.
  // The phone cannot install, so it does not ask about an active download.
  useEffect(() => {
    if (!enabled || !trackDownloads) return
    let cancelled = false
    void (async () => {
      const active = await api.getIcLoraDownloadActive()
      if (cancelled || !active.ok) return
      const { session_id, ic_lora_id } = active.data
      if (session_id && ic_lora_id) {
        resume(session_id, catalogVariantKey(ic_lora_id), active.data.progress ?? 0)
      }
    })()
    return () => { cancelled = true }
  }, [api, enabled, resume, trackDownloads])

  return {
    icLoras,
    catalogStatus,
    refresh,
    downloadIcLora,
    cancelDownload: cancel,
    downloadingKey,
    progress,
    downloadError,
  }
}

// Plain-LoRA catalog: uses /api/loras* (an isolated download session, so it never collides
// with an in-flight IC-LoRA download).

// The subset of the API client the LoRA catalog needs. Defaulting to the
// desktop-bound `ApiClient` keeps every desktop caller unchanged; the ltx-io
// recipe screen passes its ExploreRuntime `api` instead so the same hook works
// on the LAN remote host (this file must not import ltx-io, so the api is injected).
type LoraCatalogApi = Pick<
  typeof ApiClient,
  'listLoras' | 'startLoraDownload' | 'getLoraDownloadProgress' | 'getLoraDownloadActive'
>

export function useLoraCatalog(enabled: boolean, api: LoraCatalogApi = ApiClient) {
  const [loras, setLoras] = useState<LoraCatalogListItem[]>([])
  // Distinct from `loras.length === 0`: an empty list after a successful fetch
  // ("no adapter") must not look the same as a failed/pending fetch, or a catalog
  // error renders as an endless "loading" spinner with no way to retry.
  const [catalogStatus, setCatalogStatus] = useState<CatalogStatus>('loading')

  const refresh = useCallback(async (): Promise<LoraCatalogListItem[] | null> => {
    if (!enabled) return null
    setCatalogStatus((status) => (status === 'loaded' ? status : 'loading'))
    const r = await api.listLoras()
    if (r.ok) {
      setLoras(r.data.loras)
      setCatalogStatus('loaded')
      return r.data.loras
    }
    setCatalogStatus('error')
    return null
  }, [enabled, api])

  useEffect(() => { void refresh() }, [refresh])

  const getActiveSession = useCallback(async () => {
    const active = await api.getLoraDownloadActive()
    if (!active.ok || !active.data.session_id || !active.data.lora_id) return null
    return {
      sessionId: active.data.session_id,
      key: catalogVariantKey(active.data.lora_id),
      progress: active.data.progress ?? 0,
    }
  }, [api])

  const { downloadingKey, progress, downloadError, download: downloadLora, resume, cancel } = useCatalogDownload(
    (id, variantId) => api.startLoraDownload({
      lora_id: id,
      variant_id: variantId,
      use_hf_auth: true,
    }),
    (sessionId) => api.getLoraDownloadProgress({ sessionId }),
    refresh,
    getActiveSession,
  )

  // On mount, reattach to a download that's already in flight (only one plain-LoRA session
  // runs at a time) so a screen opened mid-download shows live progress instead of a stale
  // "Download" button. The session tracks only the lora id, so we key on the default variant.
  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    void (async () => {
      const active = await api.getLoraDownloadActive()
      if (cancelled || !active.ok) return
      const { session_id, lora_id } = active.data
      if (session_id && lora_id) {
        resume(session_id, catalogVariantKey(lora_id), active.data.progress ?? 0)
      }
    })()
    return () => { cancelled = true }
  }, [enabled, api, resume])

  return { loras, catalogStatus, refresh, downloadLora, cancelDownload: cancel, downloadingKey, progress, downloadError }
}
