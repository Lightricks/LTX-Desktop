import { AlertCircle, Download, Monitor, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Button } from './ui/button'

interface ModelFormat {
  id: string
  name: string
  size_gb: number
  min_vram_gb: number
  quality_tier: string
  needs_distilled_lora: boolean
  download_url: string
  description: string
}

interface DistilledLora {
  name: string
  size_gb: number
  download_url: string
  description: string
}

interface GuideData {
  gpu_name: string | null
  vram_gb: number | null
  recommended_format: string
  formats: ModelFormat[]
  distilled_lora: DistilledLora
}

interface ModelGuideDialogProps {
  isOpen: boolean
  onClose: () => void
}

const QUALITY_COLORS: Record<string, string> = {
  'Best': 'bg-green-500/20 text-green-400 border-green-500/30',
  'Excellent': 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  'Very Good': 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  'Good': 'bg-amber-500/20 text-amber-400 border-amber-500/30',
}

const SETUP_STEPS = [
  'Click the Download button on the model you want. It\'ll take you to the download page — grab the file.',
  'If the model says it needs the Speed Boost LoRA, download that too (the yellow box below).',
  'Go to Settings → Models and set the folder where you want to keep your model files.',
  'Move the downloaded files into that folder.',
  'Hit "Scan" in the Models tab — your new model will show up and you\'re ready to go.',
]

export function ModelGuideDialog({ isOpen, onClose }: ModelGuideDialogProps) {
  const [guide, setGuide] = useState<GuideData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!isOpen) return
    setLoading(true)
    setError(null)
    window.electronAPI.getBackendUrl().then((backendUrl) => {
      return fetch(`${backendUrl}/api/models/video/guide`)
    }).then((res) => {
      if (!res.ok) throw new Error(`Server returned ${res.status}`)
      return res.json() as Promise<GuideData>
    }).then((data) => {
      setGuide(data)
    }).catch((err: unknown) => {
      setError(err instanceof Error ? err.message : 'Failed to load guide')
    }).finally(() => {
      setLoading(false)
    })
  }, [isOpen])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Dialog card */}
      <div className="relative z-10 w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-zinc-900 border border-zinc-700 rounded-xl shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
          <h2 className="text-base font-semibold text-white tracking-tight">Which Model Do I Need?</h2>
          <button
            onClick={onClose}
            className="p-1.5 rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {loading && (
            <div className="flex items-center justify-center py-12">
              <div className="h-6 w-6 rounded-full border-2 border-purple-500 border-t-transparent animate-spin" />
              <span className="ml-3 text-sm text-zinc-400">Loading recommendations…</span>
            </div>
          )}

          {error && (
            <div className="flex items-center gap-2 rounded-lg bg-red-500/10 border border-red-500/30 px-4 py-3 text-sm text-red-400">
              <AlertCircle className="h-4 w-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {guide && (
            <>
              {/* GPU Banner */}
              <div className="flex items-center gap-3 rounded-lg bg-zinc-800/60 border border-zinc-700/50 px-4 py-3">
                <Monitor className="h-5 w-5 text-purple-400 flex-shrink-0" />
                <div>
                  {guide.gpu_name ? (
                    <p className="text-sm font-medium text-white">
                      {guide.gpu_name}
                      {guide.vram_gb !== null && (
                        <span className="ml-2 text-xs font-normal text-zinc-400">
                          {guide.vram_gb} GB VRAM
                        </span>
                      )}
                    </p>
                  ) : (
                    <p className="text-sm text-zinc-400">No GPU detected — API mode only</p>
                  )}
                  <p className="text-xs text-zinc-500 mt-0.5">
                    Best pick for your GPU:{' '}
                    <span className="text-purple-300 font-medium">
                      {guide.formats.find(f => f.id === guide.recommended_format)?.name ?? guide.recommended_format}
                    </span>
                  </p>
                </div>
              </div>

              {/* Format cards */}
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-1">
                  Pick a Model Size
                </h3>
                <p className="text-xs text-zinc-500 mb-3">
                  Smaller models use less video memory so they run on cheaper GPUs, but the video quality goes down a bit.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {guide.formats.map((fmt) => {
                    const isRecommended = fmt.id === guide.recommended_format
                    const qualityClass =
                      QUALITY_COLORS[fmt.quality_tier] ??
                      'bg-zinc-700/40 text-zinc-400 border-zinc-600/40'

                    return (
                      <div
                        key={fmt.id}
                        className={`relative flex flex-col gap-3 rounded-lg border p-4 transition-colors ${
                          isRecommended
                            ? 'bg-purple-500/10 border-purple-500/40'
                            : 'bg-zinc-800/40 border-zinc-700/50'
                        }`}
                      >
                        {/* Recommended pill */}
                        {isRecommended && (
                          <span className="absolute top-3 right-3 rounded-full bg-purple-600/30 border border-purple-500/40 px-2 py-0.5 text-[10px] font-semibold text-purple-300 uppercase tracking-wide">
                            Recommended
                          </span>
                        )}

                        <div className="pr-20">
                          <p className="text-sm font-semibold text-white">{fmt.name}</p>
                          <p className="text-xs text-zinc-400 mt-0.5 leading-relaxed">
                            {fmt.description}
                          </p>
                        </div>

                        <div className="flex flex-wrap items-center gap-2 text-xs">
                          {/* Quality badge */}
                          <span
                            className={`rounded-md border px-2 py-0.5 font-medium ${qualityClass}`}
                          >
                            {fmt.quality_tier}
                          </span>

                          {/* Size */}
                          <span className="text-zinc-500">{fmt.size_gb} GB</span>

                          {/* VRAM */}
                          <span className="text-zinc-500">≥ {fmt.min_vram_gb} GB VRAM</span>
                        </div>

                        <Button
                          size="sm"
                          onClick={() => window.open(fmt.download_url, '_blank')}
                          className={`w-full gap-1.5 text-xs ${
                            isRecommended
                              ? 'bg-purple-600 hover:bg-purple-700 text-white'
                              : 'bg-zinc-700 hover:bg-zinc-600 text-white'
                          }`}
                        >
                          <Download className="h-3.5 w-3.5" />
                          Download
                        </Button>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Distilled LoRA callout */}
              <div className="flex gap-3 rounded-lg bg-amber-500/10 border border-amber-500/30 p-4">
                <AlertCircle className="h-5 w-5 text-amber-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0 space-y-2">
                  <div>
                    <p className="text-sm font-semibold text-amber-300">{guide.distilled_lora.name}</p>
                    <p className="text-xs text-amber-400/80 mt-0.5 leading-relaxed">
                      {guide.distilled_lora.description}
                    </p>
                    <p className="text-xs text-amber-500/70 mt-1">
                      {guide.distilled_lora.size_gb} GB
                    </p>
                  </div>
                  <Button
                    size="sm"
                    onClick={() => window.open(guide.distilled_lora.download_url, '_blank')}
                    className="gap-1.5 text-xs bg-amber-600/30 hover:bg-amber-600/50 border border-amber-500/40 text-amber-300"
                  >
                    <Download className="h-3.5 w-3.5" />
                    Download Distilled LoRA
                  </Button>
                </div>
              </div>

              {/* Setup instructions */}
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-3">
                  How to Set It Up
                </h3>
                <ol className="space-y-2">
                  {SETUP_STEPS.map((step, i) => (
                    <li key={i} className="flex gap-3 text-xs text-zinc-400 leading-relaxed">
                      <span className="flex-shrink-0 flex items-center justify-center h-5 w-5 rounded-full bg-zinc-700 text-zinc-300 font-semibold text-[10px]">
                        {i + 1}
                      </span>
                      <span className="pt-0.5">{step}</span>
                    </li>
                  ))}
                </ol>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
