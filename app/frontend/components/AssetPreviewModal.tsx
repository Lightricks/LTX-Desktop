import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { Check, ChevronLeft, ChevronRight, Copy, X } from 'lucide-react'
import type { Asset } from '../types/project-model'
import { pathToFileUrl } from '../lib/file-url'
import { formatPipelineDisplayName } from '../lib/video-generation-model-specs'

const mediaClassName =
  'mx-auto max-h-[calc(100dvh-18vh-7rem)] max-w-full rounded-xl object-contain'

// Duration can be a raw float (e.g. extend output: 12.041667s). Show a clean value:
// integers as-is, otherwise at most 2 decimals with trailing zeros trimmed.
function formatSeconds(seconds: number): string {
  return Number.isInteger(seconds) ? String(seconds) : seconds.toFixed(2).replace(/\.?0+$/, '')
}

function formatDurationLabel(asset: Asset): string | null {
  if (asset.type === 'image') return 'Image'
  // Requested Auto is persisted as generationParams.duration === null. Do not fall through
  // to omitting the chip — that made Auto clips look like they had no duration at all.
  if (asset.generationParams?.duration === null) return 'Auto duration'
  if (asset.duration) return `${formatSeconds(asset.duration)}s`
  return null
}

export interface AssetPreviewModalProps {
  asset: Asset
  /** 0-based index into the current filtered gallery. */
  index: number
  total: number
  canGoPrev: boolean
  canGoNext: boolean
  onPrev: () => void
  onNext: () => void
  onClose: () => void
}

export function AssetPreviewModal({
  asset,
  index,
  total,
  canGoPrev,
  canGoNext,
  onPrev,
  onNext,
  onClose,
}: AssetPreviewModalProps) {
  const [copiedPrompt, setCopiedPrompt] = useState(false)

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') { e.preventDefault(); onPrev() }
      else if (e.key === 'ArrowRight') { e.preventDefault(); onNext() }
      else if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onPrev, onNext, onClose])

  // Reset copy affordance when paging between assets.
  useEffect(() => {
    setCopiedPrompt(false)
  }, [asset.id])

  // Image assets also store a placeholder `model` (e.g. "fast" for Z-Image) that is not an
  // LTX video pipeline — only label known video pipelines on video assets. Prefer the label
  // captured at generation time: the local "fast" pipeline id is shared by every LTX version,
  // so mapping it here would report the wrong version.
  const modelLabel = asset.type === 'video'
    ? asset.generationParams?.modelLabel ?? formatPipelineDisplayName(asset.generationParams?.model)
    : null
  const metaParts = [
    modelLabel,
    asset.resolution || null,
    formatDurationLabel(asset),
  ].filter(Boolean)

  return createPortal(
    <div
      className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/90"
      onClick={onClose}
    >
      <button
        onClick={(e) => { e.stopPropagation(); onPrev() }}
        disabled={!canGoPrev}
        className={`absolute left-4 top-1/2 -translate-y-1/2 z-10 p-3 rounded-full backdrop-blur-md transition-all ${
          canGoPrev
            ? 'bg-white/10 text-white hover:bg-white/20 cursor-pointer'
            : 'bg-white/5 text-zinc-600 cursor-default'
        }`}
      >
        <ChevronLeft className="h-6 w-6" />
      </button>

      <button
        onClick={(e) => { e.stopPropagation(); onNext() }}
        disabled={!canGoNext}
        className={`absolute right-4 top-1/2 -translate-y-1/2 z-10 p-3 rounded-full backdrop-blur-md transition-all ${
          canGoNext
            ? 'bg-white/10 text-white hover:bg-white/20 cursor-pointer'
            : 'bg-white/5 text-zinc-600 cursor-default'
        }`}
      >
        <ChevronRight className="h-6 w-6" />
      </button>

      <div
        className="relative flex max-h-full w-full max-w-5xl flex-col px-20 py-6"
        onClick={e => e.stopPropagation()}
      >
        <div className="mb-3 flex shrink-0 items-center justify-between">
          <span className="text-sm text-zinc-500 font-medium">
            {index + 1} / {total}
          </span>
          <button
            onClick={onClose}
            className="p-2 rounded-md text-zinc-400 hover:text-white transition-colors"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        {asset.type === 'video' ? (
          <video
            key={asset.id}
            src={pathToFileUrl(asset.path)}
            controls
            autoPlay
            className={mediaClassName}
          />
        ) : (
          <img
            key={asset.id}
            src={pathToFileUrl(asset.path)}
            alt=""
            className={mediaClassName}
          />
        )}
        <div className="mt-3 shrink-0 text-center">
          <div className="inline-flex max-w-full items-start gap-2">
            <p className="max-h-[18vh] overflow-y-auto whitespace-pre-wrap break-words text-left text-zinc-300">{asset.prompt}</p>
            {asset.prompt && (
              <button
                onClick={() => {
                  void navigator.clipboard.writeText(asset.prompt)
                  setCopiedPrompt(true)
                  setTimeout(() => setCopiedPrompt(false), 2000)
                }}
                className="shrink-0 p-1 rounded hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200 transition-colors"
                title="Copy prompt"
              >
                {copiedPrompt ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
              </button>
            )}
          </div>
          {metaParts.length > 0 && (
            <p className="mt-1 text-sm text-zinc-500">
              {metaParts.join(' • ')}
            </p>
          )}
        </div>
      </div>
    </div>,
    document.body,
  )
}
