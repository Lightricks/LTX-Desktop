import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Download, ExternalLink, Check, Loader2, Sparkles } from 'lucide-react'
import { formatBytes } from '@/lib/format'
import { catalogVariantKey, preferredVariantId } from '@/lib/lora-library'

export type LibraryItemStatus =
  | { kind: 'available' }
  | { kind: 'downloading'; progress: number }
  | { kind: 'downloaded' }
  | { kind: 'error'; gated?: boolean; message: string }

export interface LibraryItemVariant {
  id: string
  label: string
  sizeBytes: number
  downloaded?: boolean
}

export interface LibraryItem {
  id: string
  title: string
  description?: string
  sizeBytes?: number
  /** Item-level fallback when there are no variants (any file present). */
  downloaded: boolean
  thumbnailUrl?: string
  demoVideoUrl?: string
  author?: { name: string; url?: string | null }
  license?: { name: string; url?: string | null }
  variants?: LibraryItemVariant[]
  defaultVariantId?: string
  requiresHfLogin?: boolean
}

interface LibraryItemCardProps {
  item: LibraryItem
  selected?: boolean
  // The variant actually active for this entry (when selected). Undefined/null means
  // "don't check" (e.g. no variant picker) — badge follows `selected` alone.
  selectedVariantId?: string | null
  downloadingKey?: string | null
  progress?: number
  downloadError?: { key: string; message: string; gated?: boolean } | null
  onDownload?: (id: string, variantId?: string) => void
  onUse?: (id: string, variantId?: string) => void
  onRequestAccess?: (id: string) => void
  onRetry?: (id: string, variantId?: string) => void
  infoSlot?: ReactNode
}

const openUrl = (url?: string | null) => { if (url) void window.electronAPI.openExternalUrl({ url }) }

// Generic media grid card for one downloadable asset. No domain knowledge — the consumer
// maps its type into a LibraryItem. Media is optional: a demo video shows its first frame
// (preload=metadata) and plays on hover; without media the card shows the thumbnail or a placeholder.
export function LibraryItemCard({
  item, selected, selectedVariantId, downloadingKey, progress = 0, downloadError,
  onDownload, onUse, onRequestAccess, onRetry, infoSlot,
}: LibraryItemCardProps) {
  const hasVariants = (item.variants?.length ?? 0) > 1
  const downloadedVariantIds = useMemo(
    () => (item.variants ?? []).filter(v => v.downloaded).map(v => v.id),
    [item.variants],
  )
  const downloadedKey = downloadedVariantIds.join(',')
  const fallbackVariantId = item.defaultVariantId ?? item.variants?.[0]?.id
  const initialVariantId = preferredVariantId(item.variants, item.defaultVariantId, downloadedVariantIds)
    ?? fallbackVariantId
  const [variantId, setVariantId] = useState(initialVariantId)
  // Keep selection on an installed checkpoint when install state arrives after first paint
  // (e.g. listModels refresh) and the catalog default still isn't on disk.
  useEffect(() => {
    if (!hasVariants) return
    const preferred = preferredVariantId(item.variants, item.defaultVariantId, downloadedVariantIds)
    setVariantId(prev => {
      if (prev && downloadedVariantIds.includes(prev)) return prev
      if (prev && item.variants?.some(v => v.id === prev)) return prev
      return preferred ?? fallbackVariantId
    })
  }, [hasVariants, item.defaultVariantId, item.variants, downloadedKey, fallbackVariantId]) // eslint-disable-line react-hooks/exhaustive-deps -- downloadedKey proxies ids

  const selectedVariant = item.variants?.find(v => v.id === variantId)
  const sizeBytes = selectedVariant?.sizeBytes ?? item.sizeBytes
  const size = sizeBytes ? formatBytes(sizeBytes) : null
  const activeKey = catalogVariantKey(item.id, hasVariants ? variantId : undefined)
  const status: LibraryItemStatus = (() => {
    if (downloadingKey === activeKey) return { kind: 'downloading', progress }
    if (downloadError?.key === activeKey) {
      return {
        kind: 'error',
        gated: downloadError.gated ?? Boolean(item.requiresHfLogin),
        message: downloadError.message,
      }
    }
    if (hasVariants) {
      return selectedVariant?.downloaded ? { kind: 'downloaded' } : { kind: 'available' }
    }
    return item.downloaded ? { kind: 'downloaded' } : { kind: 'available' }
  })()
  const selectedVariantReady = status.kind === 'downloaded'
  // With a variant picker, only badge "Selected" when the dropdown is on the variant
  // that's actually active — not just this catalog entry (avoids marking the wrong checkpoint).
  const isSelected = Boolean(selected) && (!hasVariants || selectedVariantId == null || selectedVariantId === variantId)
  const [hovered, setHovered] = useState(false)
  // Lazy media: don't fetch/decode the demo video until the card scrolls near the viewport.
  const [near, setNear] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const cardRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = cardRef.current
    if (!el || near) return
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) setNear(true)
    }, { rootMargin: '200px' })
    observer.observe(el)
    return () => observer.disconnect()
  }, [near])

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    if (hovered) void v.play().catch(() => {})
    else { v.pause(); v.currentTime = 0 }
  }, [hovered])

  // Keep the variant picker visible while another variant downloads.
  const showVariantPicker = hasVariants && status.kind !== 'downloading'

  return (
    <div
      ref={cardRef}
      className={`flex flex-col overflow-hidden rounded-xl border transition-colors ${isSelected ? 'border-amber-500/60 bg-amber-500/10' : 'border-zinc-700 bg-zinc-800/50'}`}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* Media */}
      <div className="relative aspect-[4/3] w-full overflow-hidden bg-zinc-950">
        {item.demoVideoUrl ? (
          <video
            ref={videoRef}
            src={near || hovered ? item.demoVideoUrl : undefined}
            poster={item.thumbnailUrl}
            muted
            loop
            playsInline
            preload="metadata"
            className="h-full w-full object-cover"
          />
        ) : item.thumbnailUrl ? (
          <img src={item.thumbnailUrl} alt={item.title} className="h-full w-full object-cover" />
        ) : (
          <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-zinc-800 to-zinc-900">
            <Sparkles className="h-6 w-6 text-zinc-700" />
          </div>
        )}
        {size && (
          <span className="absolute bottom-1.5 right-1.5 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-zinc-300">{size}</span>
        )}
      </div>

      {/* Body */}
      <div className="flex flex-1 flex-col gap-1.5 p-2.5">
        <div className="flex items-center gap-1.5">
          <span className="truncate text-sm font-medium text-white" title={item.title}>{item.title}</span>
          {infoSlot}
        </div>
        {item.description && <div className="line-clamp-2 text-[11px] text-zinc-500">{item.description}</div>}
        {(item.author || item.license) && (
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[10px] text-zinc-500">
            {item.author && (
              item.author.url
                ? <button onClick={() => openUrl(item.author!.url)} className="text-zinc-400 hover:text-white transition-colors">by {item.author.name}</button>
                : <span>by {item.author.name}</span>
            )}
            {item.license && (
              item.license.url
                ? <button onClick={() => openUrl(item.license!.url)} className="rounded bg-zinc-700/60 px-1.5 py-0.5 text-zinc-300 hover:bg-zinc-700 transition-colors">{item.license.name}</button>
                : <span className="rounded bg-zinc-700/60 px-1.5 py-0.5 text-zinc-300">{item.license.name}</span>
            )}
          </div>
        )}

        {/* Footer: status / action */}
        <div className="mt-auto pt-1.5 flex flex-col gap-1.5">
          {showVariantPicker && (
            <select
              value={variantId}
              onChange={(e) => setVariantId(e.target.value)}
              className="w-full rounded bg-zinc-900 border border-zinc-700 px-2 py-1 text-[11px] text-zinc-200"
              aria-label={`${item.title} variant`}
            >
              {item.variants!.map(v => (
                <option key={v.id} value={v.id}>
                  {v.downloaded ? `${v.label} ✓` : v.label}
                </option>
              ))}
            </select>
          )}
          {status.kind === 'downloading' && (
            <div className="flex flex-col gap-1.5">
              <span className="inline-flex items-center gap-1 text-xs text-zinc-400">
                <Loader2 className="h-3 w-3 animate-spin" /> {Math.round(status.progress)}%
              </span>
              <div className="h-1.5 overflow-hidden rounded-full bg-zinc-800">
                <div className="h-full bg-blue-500 transition-all duration-300" style={{ width: `${status.progress}%` }} />
              </div>
            </div>
          )}
          {status.kind === 'error' && (
            <div className="flex flex-col gap-1.5">
              <div className="text-[11px] text-red-400" title={status.message}>
                {status.gated ? 'This model is gated — request access, then retry.' : status.message}
              </div>
              <div className="flex gap-2">
                {status.gated && (
                  <button
                    onClick={() => onRequestAccess?.(item.id)}
                    className="inline-flex items-center gap-1 rounded bg-amber-600 px-2 py-1 text-[11px] font-medium text-white transition-colors hover:bg-amber-500"
                  >
                    <ExternalLink className="h-3 w-3" /> Request access
                  </button>
                )}
                <button
                  onClick={() => onRetry?.(item.id, hasVariants ? variantId : undefined)}
                  className="inline-flex items-center gap-1 rounded bg-zinc-700 px-2 py-1 text-[11px] font-medium text-zinc-200 transition-colors hover:bg-zinc-600"
                >
                  <Download className="h-3 w-3" /> Retry
                </button>
              </div>
            </div>
          )}
          {status.kind !== 'downloading' && status.kind !== 'error' && selectedVariantReady && (
            <button
              onClick={() => onUse?.(item.id, hasVariants ? variantId : undefined)}
              className={`inline-flex w-full items-center justify-center gap-1 rounded px-3 py-1.5 text-xs font-medium transition-colors ${isSelected ? 'bg-amber-600 text-white' : 'bg-zinc-700 text-zinc-200 hover:bg-zinc-600'}`}
            >
              {isSelected ? <><Check className="h-3 w-3" /> Selected</> : 'Use'}
            </button>
          )}
          {status.kind !== 'downloading' && status.kind !== 'error' && !selectedVariantReady && (
            <button
              onClick={() => onDownload?.(item.id, hasVariants ? variantId : undefined)}
              className="inline-flex w-full items-center justify-center gap-1 rounded bg-blue-600 px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-blue-500"
            >
              <Download className="h-3 w-3" /> Download
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
