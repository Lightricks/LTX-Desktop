import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import {
  Trash2, Download, Image, Video, X,
  Heart, Film, Volume2, VolumeX, Sparkles, Sparkle,
  Clock, Monitor, ChevronUp, Scissors, Music, Undo2, Redo2, Loader2,
  ChevronLeft, ChevronRight, Copy, Check, MoveHorizontal, Wand2
} from 'lucide-react'
import { useProjects } from '../contexts/ProjectContext'
import type { GenSpaceRetakeSource } from '../contexts/ProjectContext'
import { useAppSettings } from '../contexts/AppSettingsContext'
import { useGeneration, GENERATION_RECOVERY_KEY, type GenerationRecoveryContext } from '../hooks/use-generation'
import { setActiveGenerationOwner, hasValidBaselineId } from '../lib/generation-recovery'
import { withGenerationActive } from '../lib/generation-active'
import { useVideoGenerationModelSpecs } from '../hooks/use-video-generation-model-specs'
import { createLocalGenerationError, type GenerationError } from '../lib/generation-errors'
import { useRetake } from '../hooks/use-retake'
import { useExtend, type ExtendDirection, EXTEND_SECONDS, DEFAULT_EXTEND_SECONDS } from '../hooks/use-extend'
import { resolutionOptions, type ResolutionOption } from '../lib/video-resolution'
import { useIcLora, type IcLoraAudioMode } from '../hooks/use-ic-lora'
import { useCustomIcLoraEnabled } from '../hooks/use-custom-ic-lora-enabled'
import { useDevFlags } from '../contexts/DevFlagsContext'
import { type IcLoraListItem } from '../hooks/use-catalog'
import { useIcLoraLibrary } from '../hooks/use-ic-lora-library'
import { useLoraLibrary } from '../hooks/use-lora-library'
import { usePromptEnhancerProvider } from '../hooks/use-prompt-enhancer-provider'
import { useGlobalGenerationLock } from '../hooks/use-global-generation-lock'
import type { ICLoraConditioningType } from '../components/ICLoraPanel'
import type { Asset } from '../types/project-model'
import { GenerationErrorDialog } from '../components/GenerationErrorDialog'
import { addVisualAssetToProject } from '../lib/asset-copy'
import { pathToFileUrl } from '../lib/file-url'
import {
  areVideoGenerationSettingsEquivalent,
  getVideoGenerationModelSpecs,
  resolveVideoGenerationOptions,
  sanitizeVideoGenerationSettings,
  type VideoGenerationModelSpecItem,
} from '../lib/video-generation-model-specs'
import { logger } from '../lib/logger'
import { ApiClient, type ApiSuccessOf } from '../lib/api-client'
import type { LoraSelection } from '../components/SettingsPanel'
import { RetakePanel } from '../components/RetakePanel'
import { ExtendPanel } from '../components/ExtendPanel'
import { ICLoraPanel, CONDITIONING_TYPES } from '../components/ICLoraPanel'
import type { OutpaintPads } from '../components/OutpaintCanvasEditor'
import { LoraLibraryModal } from '../components/LoraLibraryModal'
import { SelectedLoraInfo } from '../components/SelectedLoraInfo'
import { buildIcLoraSelectorOptions, encodeIcLoraSelectorValue, parseIcLoraSelectorValue, toModelsDirRelativeRef, variantDisplayName } from '../lib/lora-library'
import { SettingsDropdown } from '../components/SettingsDropdown'
import { IcLoraSettingsControls, type IcLoraControlsProps } from '../components/IcLoraSettingsControls'
import { IcLoraAdvancedPanel } from '../components/IcLoraAdvancedPanel'
import { FreeApiKeyBubble } from '../components/FreeApiKeyBubble'

// Asset card with hover overlays
function AssetCard({
  asset,
  onDelete,
  onPlay,
  onDragStart,
  onCreateVideo,
  onEditImage,
  onRetake,
  onExtend,
  onIcLora,
  onToggleFavorite
}: {
  asset: Asset
  onDelete: () => void
  onPlay: () => void
  onDragStart: (e: React.DragEvent, asset: Asset) => void
  onCreateVideo?: (asset: Asset) => void
  onEditImage?: (asset: Asset) => void
  onRetake?: (asset: Asset) => void
  onExtend?: (asset: Asset) => void
  onIcLora?: (asset: Asset) => void
  onToggleFavorite?: () => void
}) {
  const hoverVideoRef = useRef<HTMLVideoElement>(null)
  const [isHovered, setIsHovered] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [isMuted, setIsMuted] = useState(true)
  const [volume, setVolume] = useState(0.5)
  const isFavorite = asset.favorite || false

  useEffect(() => {
    if (asset.type !== 'video') return
    if (!isHovered) {
      setCurrentTime(0)
      return
    }
    if (hoverVideoRef.current) {
      hoverVideoRef.current.muted = isMuted
      hoverVideoRef.current.volume = volume
      hoverVideoRef.current.play().catch(() => {})
    }
  }, [asset.type, isHovered, isMuted, volume])

  const handleTimeUpdate = () => {
    if (hoverVideoRef.current) {
      setCurrentTime(hoverVideoRef.current.currentTime)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  const handleDownload = (e: React.MouseEvent) => {
    e.stopPropagation()
    const a = document.createElement('a')
    a.href = pathToFileUrl(asset.path)
    a.download = asset.path.split('/').pop() || `${asset.type}-${asset.id}`
    a.click()
  }

  return (
    <div
      className="relative group cursor-pointer rounded-xl overflow-hidden bg-zinc-900"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => {
        setIsHovered(false)
        setCurrentTime(0)
      }}
      onClick={onPlay}
      draggable={asset.type === 'image'}
      onDragStart={(e) => asset.type === 'image' && onDragStart(e, asset)}
    >
      {asset.type === 'video' ? (
        <div className="relative w-full aspect-video bg-zinc-900">
          {asset.bigThumbnailPath && (
            <img
              src={pathToFileUrl(asset.bigThumbnailPath)}
              alt=""
              className={`absolute inset-0 w-full h-full object-contain transition-opacity duration-150 ${
                isHovered ? 'opacity-0' : 'opacity-100'
              }`}
            />
          )}
          {isHovered && (
            <video
              ref={hoverVideoRef}
              src={pathToFileUrl(asset.path)}
              className="absolute inset-0 w-full h-full object-contain"
              muted={isMuted}
              loop
              autoPlay
              playsInline
              preload="metadata"
              onTimeUpdate={handleTimeUpdate}
            />
          )}
        </div>
      ) : (
        <img src={pathToFileUrl(asset.path)} alt="" className="w-full aspect-video object-contain" />
      )}
      
      {/* Favorite heart - always visible when favorited */}
      {isFavorite && !isHovered && (
        <button
          onClick={(e) => { e.stopPropagation(); onToggleFavorite?.() }}
          className="absolute top-2 left-2 p-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white transition-colors z-10"
        >
          <Heart className="h-3.5 w-3.5 fill-current" />
        </button>
      )}
      
      {/* Hover overlay */}
      <div className={`absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-black/30 transition-opacity duration-200 ${
        isHovered ? 'opacity-100' : 'opacity-0'
      }`}>
        {/* Top buttons */}
        <div className="absolute top-2 left-2 right-2 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <button
              onClick={(e) => { e.stopPropagation(); onToggleFavorite?.() }}
              className={`p-1.5 rounded-lg backdrop-blur-md transition-colors ${
                isFavorite ? 'bg-white/20 text-white' : 'bg-black/40 text-white hover:bg-black/60'
              }`}
            >
              <Heart className={`h-3.5 w-3.5 ${isFavorite ? 'fill-current' : ''}`} />
            </button>
            
            {asset.type === 'image' && (
              <>
                <button
                  onClick={(e) => { e.stopPropagation(); onCreateVideo?.(asset) }}
                  className="px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors flex items-center gap-1.5 text-xs font-medium whitespace-nowrap"
                >
                  <Film className="h-3 w-3" />
                  Create video
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); onEditImage?.(asset) }}
                  className="px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors flex items-center gap-1.5 text-xs font-medium whitespace-nowrap"
                >
                  <Wand2 className="h-3 w-3" />
                  Edit image
                </button>
              </>
            )}
            {asset.type === 'video' && (
              <>
                <button
                  onClick={(e) => { e.stopPropagation(); onRetake?.(asset) }}
                  className="px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors flex items-center gap-1.5 text-xs font-medium whitespace-nowrap"
                >
                  <Scissors className="h-3 w-3" />
                  Retake
                </button>
                {onExtend && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onExtend(asset) }}
                    className="px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors flex items-center gap-1.5 text-xs font-medium whitespace-nowrap"
                  >
                    <MoveHorizontal className="h-3 w-3" />
                    Extend
                  </button>
                )}
                {onIcLora && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onIcLora(asset) }}
                    className="px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors flex items-center gap-1.5 text-xs font-medium whitespace-nowrap"
                  >
                    <Sparkles className="h-3 w-3" />
                    IC-LoRA
                  </button>
                )}
              </>
            )}
          </div>
          
          <div className="flex items-center gap-1.5">
            <button
              onClick={handleDownload}
              className="p-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors"
            >
              <Download className="h-3.5 w-3.5" />
            </button>
            {/* Tools button hidden for now */}
          </div>
        </div>
        
        {/* Bottom controls for video */}
        {asset.type === 'video' && (
          <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <div className="px-2 py-1 rounded-lg bg-black/50 backdrop-blur-md text-white text-xs font-mono">
                {formatTime(currentTime)}
              </div>
              <div className="flex items-center gap-1.5 rounded-lg bg-black/40 backdrop-blur-md pl-1.5 pr-2 py-1">
                <button
                  onClick={(e) => { e.stopPropagation(); setIsMuted(!isMuted) }}
                  className="text-white hover:text-white/80 transition-colors"
                  aria-label={isMuted ? 'Unmute' : 'Mute'}
                >
                  {isMuted ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
                </button>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={isMuted ? 0 : volume}
                  onClick={(e) => e.stopPropagation()}
                  onMouseDown={(e) => e.stopPropagation()}
                  onChange={(e) => {
                    e.stopPropagation()
                    const next = parseFloat(e.target.value)
                    setVolume(next)
                    if (next === 0) {
                      setIsMuted(true)
                    } else if (isMuted) {
                      setIsMuted(false)
                    }
                  }}
                  className="w-16 h-1 accent-white cursor-pointer"
                  aria-label="Volume"
                />
              </div>
            </div>
          </div>
        )}

        {/* Delete button (subtle, bottom right) */}
        {(
          <button
            onClick={(e) => { e.stopPropagation(); onDelete() }}
            className="absolute bottom-2 right-2 p-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white/70 hover:bg-red-500/80 hover:text-white transition-colors opacity-0 group-hover:opacity-100"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
      
    </div>
  )
}

// Lightricks brand icon
function LightricksIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path fillRule="evenodd" clipRule="evenodd" d="M17.0073 8.18934C16.3266 5.6556 14.9346 2.06903 12.3065 2.06903C9.27204 2.06903 6.86627 7.24621 5.45487 11.7948C4.79654 13.9203 4.35877 15.9049 4.17755 17.1736C4.10214 17.5829 4.06274 18.0044 4.06274 18.4347C4.06274 22.2903 7.22553 25.4338 11.1133 25.4338C15.5206 25.4338 23.9376 22.7073 23.9376 18.4347C23.9376 17.1179 23.1376 15.948 21.9018 14.9595L21.9039 14.9575C22.4493 13.7707 22.847 12.648 23.001 11.705C23.1934 10.5053 23.0074 9.5494 22.4429 8.88217C21.7692 8.07382 20.7107 7.85572 19.6586 7.84288C18.8826 7.84288 17.9777 7.96904 17.0073 8.18934ZM8.00176 9.17083C7.6945 9.93266 7.02317 11.7419 6.70157 12.9799C7.93005 11.9987 9.2965 11.1653 10.7091 10.4796C12.2325 9.73758 13.9171 9.06448 15.518 8.58411C15.08 6.98293 13.9585 3.62158 12.3129 3.62158C11.0298 3.62158 9.41958 5.69374 8.00176 9.17083ZM20.6201 14.083L20.6209 14.0786C21.0507 13.1163 21.3522 12.2118 21.4741 11.4547C21.5511 10.9607 21.5832 10.2872 21.2752 9.89577C20.9416 9.46599 20.1975 9.39543 19.6521 9.38901C18.9932 9.38901 18.2117 9.49943 17.3641 9.69208L17.3683 9.69702C17.586 10.7217 17.7526 11.772 17.8808 12.7968C18.8527 13.16 19.7877 13.5908 20.6201 14.083ZM15.8828 10.0897C14.6739 10.4588 13.4041 10.9464 12.209 11.4846C13.4346 11.588 14.8471 11.8527 16.2581 12.2608C16.1554 11.5367 16.0273 10.8061 15.8799 10.0948L15.8828 10.0897ZM11.1133 12.9816C8.07878 12.9816 5.60884 15.4258 5.60884 18.4347C5.60884 21.4435 8.07878 23.8878 11.1133 23.8878C13.8701 23.8878 16.3653 21.6639 16.6048 18.9158C16.7011 17.7546 16.669 15.9263 16.4637 13.9311C14.6294 13.3385 12.6763 12.9816 11.1133 12.9816ZM18.3883 22.2069C17.7984 22.4697 17.1711 22.7085 16.5284 22.9184C18.0872 21.3274 19.8832 18.8193 21.1982 16.3689L21.1997 16.3654C21.9756 17.0509 22.3915 17.7593 22.3915 18.4347C22.3915 19.6985 20.9288 21.0778 18.3883 22.2069ZM19.9493 15.4655L19.9473 15.4707C19.4291 16.4567 18.8221 17.4625 18.1833 18.4092C18.2214 17.4089 18.1892 16.0386 18.0611 14.5212C18.71 14.7948 19.3456 15.1021 19.9493 15.4655Z" fill="currentColor" />
    </svg>
  )
}

function ZitIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M19.113 12.2515H16.5605L14.008 8.63382L6.04545 19.9068H8.60348L14.0079 12.2518L16.5605 12.2515L11.156 19.9068H13.721L19.113 12.2515V15.8693L16.2716 19.9073V22.0063H2L14.008 5L19.113 12.2515Z" fill="currentColor"/>
      <path d="M26 22.0064L21.9704 22.0063V19.9151L19.113 15.8693V12.2515L26 22.0064Z" fill="currentColor"/>
    </svg>
  )
}

// Square icon for aspect ratio
function AspectIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="5" width="18" height="14" rx="2" />
    </svg>
  )
}

// Duration can be a raw float (e.g. extend output: 12.041667s). Show a clean value:
// integers as-is, otherwise at most 2 decimals with trailing zeros trimmed.
function formatSeconds(seconds: number): string {
  return Number.isInteger(seconds) ? String(seconds) : seconds.toFixed(2).replace(/\.?0+$/, '')
}

const DEFAULT_LORA_SCALE = 1.0
const IMAGE_STEPS_GENERATE = 4
const IMAGE_STEPS_EDIT = 8

// Multi-select LoRA picker with a per-LoRA strength slider.
function LoRAPicker({
  available,
  selected,
  onChange,
  displayNames,
  catalogIdsByPath,
}: {
  available: ApiSuccessOf<'listModels'>['models']
  selected: LoraSelection[]
  onChange: (loras: LoraSelection[]) => void
  // Catalog display names keyed by installed path (cross-checked against the catalog), so a
  // downloaded catalog LoRA shows its proper name instead of the raw filename.
  displayNames?: Map<string, string>
  // Catalog id keyed by installed path, same cross-check — attached to a new selection so the
  // prompt enhancer can look up this LoRA's trigger/instructions later. Without it, a LoRA
  // picked from this dropdown (rather than the "Browse LoRAs" catalog modal) would silently
  // enhance with no catalog awareness at all.
  catalogIdsByPath?: Map<string, string>
}) {
  const [isOpen, setIsOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!isOpen) return
    const onClick = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) setIsOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [isOpen])

  const nameFor = (lora: ApiSuccessOf<'listModels'>['models'][0]) => displayNames?.get(lora.path) ?? lora.name

  const toggle = (lora: ApiSuccessOf<'listModels'>['models'][0]) => {
    if (selected.find(s => s.ref === lora.path)) {
      onChange(selected.filter(s => s.ref !== lora.path))
    } else {
      onChange([...selected, {
        ref: lora.path,
        name: nameFor(lora),
        scale: DEFAULT_LORA_SCALE,
        catalogId: catalogIdsByPath?.get(lora.path),
      }])
    }
  }

  const updateScale = (loraRef: string, scale: number) => {
    onChange(selected.map(s => s.ref === loraRef ? { ...s, scale } : s))
  }

  const label = selected.length === 0 ? 'LoRA' : selected.length === 1 ? selected[0].name : `${selected.length} LoRAs`

  return (
    <div ref={containerRef} className="relative">
      <button
        onClick={() => setIsOpen(o => !o)}
        className="flex items-center gap-1.5 px-2 py-1.5 rounded-md bg-zinc-800/60 text-zinc-300 text-xs hover:bg-zinc-700/60 transition-colors max-w-[160px]"
      >
        <Sparkles className="h-3.5 w-3.5 flex-shrink-0" />
        <span className="truncate">{label}</span>
      </button>
      {isOpen && (
        <div className="absolute bottom-full mb-1 left-0 z-50 w-72 max-h-80 overflow-y-auto rounded-md border border-zinc-700 bg-zinc-900 shadow-xl">
          <div className="px-3 py-2 text-[10px] uppercase tracking-wide text-zinc-500 border-b border-zinc-800">
            Select LoRAs
          </div>
          {available.map(lora => {
            const sel = selected.find(s => s.ref === lora.path)
            return (
              <div key={lora.path} className="px-3 py-2 hover:bg-zinc-800 transition-colors">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={Boolean(sel)}
                    onChange={() => toggle(lora)}
                    className="accent-white"
                  />
                  <span className="text-xs text-zinc-300 truncate flex-1" title={nameFor(lora)}>{nameFor(lora)}</span>
                </div>
                {sel && (
                  <div className="flex items-center gap-2 mt-1.5 pl-6">
                    <input
                      type="range"
                      min={0}
                      max={2}
                      step={0.05}
                      value={sel.scale}
                      onChange={e => updateScale(lora.path, parseFloat(e.target.value))}
                      className="flex-1 accent-white"
                    />
                    <span className="text-[10px] text-zinc-400 w-8 text-right">{sel.scale.toFixed(2)}</span>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

type GenSpaceMode = 'image' | 'video' | 'retake' | 'extend' | 'ic-lora'

// Resolve a selected resolution option key to {width,height}, or undefined for "original"
// (backend then uses the source resolution).
function resolveResolution(options: ResolutionOption[], key: string): { width: number; height: number } | undefined {
  const opt = options.find((o) => o.key === key)
  if (!opt || opt.width == null || opt.height == null) return undefined
  return { width: opt.width, height: opt.height }
}

// Prompt bar component matching the design
// Two-row layout: prompt row on top, settings row below
function PromptBar({
  mode,
  onModeChange,
  canUseIcLora,
  prompt,
  onPromptChange,
  onGenerate,
  isGenerating,
  inputImage,
  onInputImageChange,
  inputAudio,
  onInputAudioChange,
  settings,
  onSettingsChange,
  videoModelSpecs,
  videoSettingsMessage,
  canGenerate,
  buttonLabel,
  buttonIcon,
  extendDirection,
  onExtendDirectionChange,
  extendSeconds,
  onExtendSecondsChange,
  resolutionOptions: resolutionOpts,
  selectedResolution,
  onResolutionChange,
  icLoraControls,
  promptOptional,
  isLocalMode,
  imageUsesFalApi,
  availableLoras,
  selectedLoras,
  onSelectedLorasChange,
  loraDisplayNames,
  loraCatalogIdsByPath,
  enhanceProviderAvailable,
  enhanceAvailableForMode,
  canEnhancePrompt,
  isEnhancingPrompt,
  enhancePromptError,
  onEnhancePrompt,
  enhanceProvider,
  onEnhanceProviderChange,
  canUndoPrompt,
  onUndoPrompt,
  canRedoPrompt,
  onRedoPrompt,
}: {
  mode: GenSpaceMode
  onModeChange: (mode: GenSpaceMode) => void
  canUseIcLora: boolean
  prompt: string
  onPromptChange: (prompt: string) => void
  onGenerate: () => void
  isGenerating: boolean
  canGenerate: boolean
  buttonLabel: string
  buttonIcon: React.ReactNode
  extendDirection?: ExtendDirection
  onExtendDirectionChange?: (direction: ExtendDirection) => void
  extendSeconds?: number
  onExtendSecondsChange?: (seconds: number) => void
  resolutionOptions?: ResolutionOption[]
  selectedResolution?: string
  onResolutionChange?: (key: string) => void
  inputImage: string | null
  onInputImageChange: (path: string | null) => void
  inputAudio: string | null
  onInputAudioChange: (path: string | null) => void
  settings: {
    model: string
    duration: number
    videoResolution: string
    fps: number
    aspectRatio: string
    imageResolution: string
    variations: number
    audio?: boolean
    imageEditStrength?: number
  }
  onSettingsChange: (settings: any) => void
  videoModelSpecs: VideoGenerationModelSpecItem[]
  videoSettingsMessage?: string | null
  icLoraControls?: IcLoraControlsProps
  promptOptional?: boolean
  isLocalMode?: boolean
  imageUsesFalApi?: boolean
  availableLoras?: ApiSuccessOf<'listModels'>['models']
  selectedLoras?: LoraSelection[]
  onSelectedLorasChange?: (loras: LoraSelection[]) => void
  loraDisplayNames?: Map<string, string>
  loraCatalogIdsByPath?: Map<string, string>
  // Whether at least one enhancer provider (local Gemma text encoder or Gemini API) is usable.
  enhanceProviderAvailable?: boolean
  // Whether the current mode supports Enhance at all (independent of prompt text / in-flight
  // generation) — gates the whole Enhance/Undo/Redo cluster's visibility.
  enhanceAvailableForMode?: boolean
  canEnhancePrompt?: boolean
  isEnhancingPrompt?: boolean
  enhancePromptError?: string | null
  onEnhancePrompt?: () => void
  // The dropdown is only rendered when this is passed — the parent omits it (locking the
  // button to whichever single option works) unless both a local Gemma checkpoint and a
  // Gemini API key are available.
  enhanceProvider?: 'local' | 'api'
  onEnhanceProviderChange?: (provider: 'local' | 'api') => void
  canUndoPrompt?: boolean
  onUndoPrompt?: () => void
  canRedoPrompt?: boolean
  onRedoPrompt?: () => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const audioInputRef = useRef<HTMLInputElement>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [isAudioDragOver, setIsAudioDragOver] = useState(false)
  const isRetake = mode === 'retake'
  const isExtend = mode === 'extend'
  const isIcLora = mode === 'ic-lora'
  const isEditingImage = mode === 'image' && !!inputImage

  // Resolution selector: local only, and only when there's a lower tier to pick.
  const showResolution = isLocalMode && !!resolutionOpts && resolutionOpts.length > 1
  const resolutionControl = showResolution ? (
    <SettingsDropdown
      title="RESOLUTION"
      value={selectedResolution ?? 'original'}
      onChange={(v) => onResolutionChange?.(v)}
      options={resolutionOpts!.map((o) => ({ value: o.key, label: o.label }))}
      trigger={
        <>
          <Monitor className="h-3.5 w-3.5" />
          <span>{(resolutionOpts!.find((o) => o.key === selectedResolution)?.label ?? 'Original').split(' ')[0]}</span>
          <ChevronUp className="h-3 w-3 text-zinc-500" />
        </>
      }
    />
  ) : null
  const resolvedVideoOptions = mode === 'video'
    ? resolveVideoGenerationOptions({
        settings,
        modelSpecs: videoModelSpecs,
        hasAudio: Boolean(inputAudio),
      })
    : null
  const showVideoFpsControl = Boolean(
    resolvedVideoOptions
    && resolvedVideoOptions.hasCompatibleOptions
    && resolvedVideoOptions.fpsOptions.length > 1,
  )

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const assetData = e.dataTransfer.getData('asset')
    if (assetData) {
      const asset = JSON.parse(assetData) as Asset
      if (asset.type === 'image') {
        onInputImageChange(asset.path)
      }
      return
    }

    // File drop from the OS (Finder/Explorer) — mirrors handleFileSelect.
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('image/')) {
      const filePath = window.electronAPI?.getPathForFile(file)
      onInputImageChange(filePath || URL.createObjectURL(file))
    }
  }

  const handleAudioDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsAudioDragOver(false)

    const assetData = e.dataTransfer.getData('asset')
    if (assetData) {
      const asset = JSON.parse(assetData) as Asset
      if (asset.type === 'audio') {
        onInputAudioChange(asset.path)
      }
    }

    // Handle file drops
    const file = e.dataTransfer.files?.[0]
    if (file) {
      const ext = file.name.split('.').pop()?.toLowerCase()
      if (['mp3', 'wav', 'ogg', 'aac', 'flac', 'm4a'].includes(ext || '')) {
        const filePath = window.electronAPI?.getPathForFile(file)
        if (filePath) {
          onInputAudioChange(filePath)
        }
      }
    }
  }

  const handleAudioFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const filePath = window.electronAPI?.getPathForFile(file)
      if (filePath) {
        onInputAudioChange(filePath)
      }
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith('image/')) {
      const filePath = window.electronAPI?.getPathForFile(file)
      if (filePath) {
        onInputImageChange(filePath)
      } else {
        const url = URL.createObjectURL(file)
        onInputImageChange(url)
      }
    }
  }
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isGenerating && canGenerate && !isEnhancingPrompt) {
      e.preventDefault()
      onGenerate()
    }
  }

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-2xl overflow-visible">
      {/* Top row: Image ref | Prompt | Generate */}
      <div className="flex items-start">
        {/* Input image drop zone — video mode (I2V) or image mode (edit source) */}
        {(mode === 'video' || mode === 'image') && !isRetake && !isIcLora && (
          <div
            className={`relative w-10 h-10 mx-2 mt-2 rounded-lg border-2 border-dashed transition-colors flex items-center justify-center flex-shrink-0 cursor-pointer ${
              isDragOver ? 'border-blue-500 bg-blue-500/10' : 'border-zinc-700 hover:border-zinc-500'
            }`}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
          >
            {inputImage ? (
              <>
                <img src={pathToFileUrl(inputImage)} alt="" className="w-full h-full object-cover rounded-md" />
                <button
                  onClick={(e) => { e.stopPropagation(); onInputImageChange(null) }}
                  className="absolute -top-1 -right-1 p-0.5 rounded-full bg-zinc-800 text-zinc-400 hover:text-white z-10"
                >
                  <X className="h-3 w-3" />
                </button>
              </>
            ) : (
              <Image className="h-4 w-4 text-zinc-500" />
            )}
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>
        )}

        {/* Audio drop zone — only in video mode */}
        {mode === 'video' && !isRetake && !isIcLora && (
          <div
            className={`relative w-10 h-10 mt-2 rounded-lg border-2 border-dashed transition-colors flex items-center justify-center flex-shrink-0 cursor-pointer ${
              isAudioDragOver ? 'border-emerald-500 bg-emerald-500/10' : inputAudio ? 'border-emerald-600' : 'border-zinc-700 hover:border-zinc-500'
            }`}
            onDragOver={(e) => { e.preventDefault(); setIsAudioDragOver(true) }}
            onDragLeave={() => setIsAudioDragOver(false)}
            onDrop={handleAudioDrop}
            onClick={() => audioInputRef.current?.click()}
            title={inputAudio ? 'Audio attached — click to change' : 'Attach audio for A2V'}
          >
            {inputAudio ? (
              <>
                <Music className="h-4 w-4 text-emerald-400" />
                <button
                  onClick={(e) => { e.stopPropagation(); onInputAudioChange(null) }}
                  className="absolute -top-1 -right-1 p-0.5 rounded-full bg-zinc-800 text-zinc-400 hover:text-white z-10"
                >
                  <X className="h-3 w-3" />
                </button>
              </>
            ) : (
              <Music className="h-4 w-4 text-zinc-500" />
            )}
            <input
              ref={audioInputRef}
              type="file"
              accept=".mp3,.wav,.ogg,.aac,.flac,.m4a"
              onChange={handleAudioFileSelect}
              className="hidden"
            />
          </div>
        )}

        {/* Prompt input - fills remaining width */}
        <div className="flex-1 min-w-0 py-1">
          <textarea
            value={prompt}
            onChange={(e) => onPromptChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={mode === 'retake'
              ? "Describe what should happen in the selected section..."
              : mode === 'extend'
                ? "Describe what should happen in the new frames... (optional)"
              : mode === 'ic-lora'
                ? (promptOptional
                    ? "Describe the new area, or leave empty to extend the scene..."
                    : "Describe the style or transformation to apply...")
              : mode === 'image'
                ? (isEditingImage
                    ? "Describe the change, e.g. make it photorealistic..."
                    : "A close-up of a woman talking on the phone...")
                : "The woman sips from a cup of coffee..."
            }
            className="w-full bg-transparent text-white text-sm placeholder:text-zinc-500 focus:outline-none px-2 py-2 resize-none overflow-y-auto h-[70px] leading-5"
          />
        </div>

      </div>
      
      {/* Bottom row: Mode selector + Settings */}
      <div className="flex items-center gap-0.5 px-1.5 py-1.5 border-t border-zinc-800/60 text-xs text-zinc-400">
        {/* Mode dropdown */}
        <SettingsDropdown
          title="MODE"
          value={mode}
          onChange={(v) => onModeChange(v as GenSpaceMode)}
          options={[
            { value: 'image', label: 'Generate Images', icon: <Image className="h-4 w-4" /> },
            { value: 'video', label: 'Generate Videos', icon: <Video className="h-4 w-4" /> },
            { value: 'retake', label: 'Retake', icon: <Scissors className="h-4 w-4" /> },
            { value: 'extend', label: 'Extend', icon: <MoveHorizontal className="h-4 w-4" /> },
            ...(canUseIcLora ? [{ value: 'ic-lora', label: 'IC-LoRA', icon: <Sparkles className="h-4 w-4" /> }] : []),
          ]}
          trigger={
            <>
              {mode === 'image' ? <Image className="h-3.5 w-3.5" /> : mode === 'retake' ? <Scissors className="h-3.5 w-3.5" /> : mode === 'extend' ? <MoveHorizontal className="h-3.5 w-3.5" /> : mode === 'ic-lora' ? <Sparkles className="h-3.5 w-3.5" /> : <Video className="h-3.5 w-3.5" />}
              <span className="text-zinc-300 font-medium">{mode === 'image' ? 'Image' : mode === 'retake' ? 'Retake' : mode === 'extend' ? 'Extend' : mode === 'ic-lora' ? 'IC-LoRA' : 'Video'}</span>
              <ChevronUp className="h-3 w-3 text-zinc-500" />
            </>
          }
        />
        
        <div className="flex-1" />
        
        {isRetake ? (
          resolutionControl ?? <div className="text-[10px] text-zinc-500 pr-2">Trim in the panel above, then retake</div>
        ) : isExtend ? (
          <>
            <SettingsDropdown
              title="DIRECTION"
              value={extendDirection ?? 'end'}
              onChange={(v) => onExtendDirectionChange?.(v as ExtendDirection)}
              options={[
                { value: 'end', label: 'At end' },
                { value: 'start', label: 'At start' },
              ]}
              trigger={
                <>
                  <MoveHorizontal className="h-3.5 w-3.5" />
                  <span>{extendDirection === 'start' ? 'At start' : 'At end'}</span>
                  <ChevronUp className="h-3 w-3 text-zinc-500" />
                </>
              }
            />
            <div className="w-px h-4 bg-zinc-700 mx-0.5" />
            <SettingsDropdown
              title="SECONDS TO ADD"
              value={String(extendSeconds ?? DEFAULT_EXTEND_SECONDS)}
              onChange={(v) => onExtendSecondsChange?.(parseInt(v, 10))}
              options={EXTEND_SECONDS.map((s) => ({ value: String(s), label: `${s}s` }))}
              trigger={
                <>
                  <Clock className="h-3.5 w-3.5" />
                  <span>{extendSeconds ?? DEFAULT_EXTEND_SECONDS}s</span>
                  <ChevronUp className="h-3 w-3 text-zinc-500" />
                </>
              }
            />
            {resolutionControl && (
              <>
                <div className="w-px h-4 bg-zinc-700 mx-0.5" />
                {resolutionControl}
              </>
            )}
          </>
        ) : isIcLora ? (
        <IcLoraSettingsControls {...icLoraControls} />
        ) : mode === 'image' ? (
          <>
            {/* Model indicator */}
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-zinc-800/50">
              <ZitIcon className="h-3.5 w-3.5" />
              <span className="text-zinc-300 font-medium">Z-Image Turbo{imageUsesFalApi ? ' (API)' : ''}</span>
            </div>

            {isEditingImage ? (
              // Edit runs at the source image's resolution — resolution/ratio don't apply.
              <div className="flex items-center gap-1.5 px-2 text-[10px] text-zinc-400">
                <span>Strength</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={settings.imageEditStrength ?? 0.6}
                  onChange={(e) => onSettingsChange({ ...settings, imageEditStrength: parseFloat(e.target.value) })}
                  className="w-20 accent-white"
                />
                <span className="w-8 text-right">{(settings.imageEditStrength ?? 0.6).toFixed(2)}</span>
              </div>
            ) : (
              <>
                {/* Resolution dropdown */}
                <SettingsDropdown
                  title="IMAGE RESOLUTION"
                  value={settings.imageResolution}
                  onChange={(v) => onSettingsChange({ ...settings, imageResolution: v })}
                  options={[
                    { value: '1080p', label: '1080p' },
                    { value: '1440p', label: '1440p' },
                    { value: '2048p', label: '2048p' },
                  ]}
                  trigger={
                    <>
                      <Monitor className="h-3.5 w-3.5" />
                      <span>{settings.imageResolution.replace('p', '')}</span>
                    </>
                  }
                />

                {/* Aspect ratio dropdown */}
                <SettingsDropdown
                  title="RATIO"
                  value={settings.aspectRatio}
                  onChange={(v) => onSettingsChange({ ...settings, aspectRatio: v })}
                  options={[
                    { value: '16:9', label: '16:9' },
                    { value: '1:1', label: '1:1' },
                    { value: '9:16', label: '9:16' },
                  ]}
                  trigger={
                    <>
                      <AspectIcon className="h-3.5 w-3.5" />
                      <span>{settings.aspectRatio}</span>
                    </>
                  }
                />
              </>
            )}
          </>
        ) : (
          <>
            {resolvedVideoOptions && resolvedVideoOptions.hasCompatibleOptions ? (
              <>
                <SettingsDropdown
                  title="MODEL"
                  value={resolvedVideoOptions.selectedModel ?? settings.model}
                  onChange={(v) => onSettingsChange({ ...settings, model: v })}
                  options={resolvedVideoOptions.modelOptions.map((item) => ({
                    value: item.pipeline,
                    label: item.spec.display_name,
                  }))}
                  trigger={
                    <>
                      <LightricksIcon className="h-3.5 w-3.5" />
                      <span className="text-zinc-300 font-medium">
                        {resolvedVideoOptions.modelOptions.find((item) => item.pipeline === resolvedVideoOptions.selectedModel)?.spec.display_name
                          ?? settings.model}
                      </span>
                    </>
                  }
                />

                <div className="w-px h-4 bg-zinc-700 mx-0.5" />

                <SettingsDropdown
                  title="DURATION"
                  value={String(resolvedVideoOptions.selectedDuration ?? settings.duration)}
                  onChange={(v) => onSettingsChange({ ...settings, duration: parseInt(v) })}
                  options={resolvedVideoOptions.durationOptions.map((value) => ({ value: String(value), label: `${value} Sec` }))}
                  trigger={
                    <>
                      <Clock className="h-3.5 w-3.5" />
                      <span>{resolvedVideoOptions.selectedDuration ?? settings.duration}s</span>
                    </>
                  }
                />

                <SettingsDropdown
                  title="RESOLUTION"
                  value={resolvedVideoOptions.selectedResolution ?? settings.videoResolution}
                  onChange={(v) => onSettingsChange({ ...settings, videoResolution: v })}
                  options={resolvedVideoOptions.resolutionOptions.map((value) => ({ value, label: value }))}
                  trigger={
                    <>
                      <Monitor className="h-3.5 w-3.5" />
                      <span>{(resolvedVideoOptions.selectedResolution ?? settings.videoResolution).replace('p', '')}</span>
                    </>
                  }
                />

                {showVideoFpsControl && (
                  <SettingsDropdown
                    title="FPS"
                    value={String(resolvedVideoOptions.selectedFps ?? settings.fps)}
                    onChange={(v) => onSettingsChange({ ...settings, fps: parseInt(v) })}
                    options={resolvedVideoOptions.fpsOptions.map((value) => ({ value: String(value), label: `${value}` }))}
                    trigger={
                      <>
                        <Film className="h-3.5 w-3.5" />
                        <span>{resolvedVideoOptions.selectedFps ?? settings.fps} FPS</span>
                      </>
                    }
                  />
                )}

                <SettingsDropdown
                  title="ASPECT RATIO"
                  value={settings.aspectRatio}
                  onChange={(v) => onSettingsChange({ ...settings, aspectRatio: v })}
                  options={[
                    { value: '16:9', label: '16:9' },
                    { value: '9:16', label: '9:16' },
                  ]}
                  trigger={
                    <>
                      <AspectIcon className="h-3.5 w-3.5" />
                      <span>{settings.aspectRatio}</span>
                    </>
                  }
                />

                {isLocalMode && availableLoras && availableLoras.length > 0 && (
                  <LoRAPicker
                    available={availableLoras}
                    selected={selectedLoras ?? []}
                    onChange={onSelectedLorasChange ?? (() => {})}
                    displayNames={loraDisplayNames}
                    catalogIdsByPath={loraCatalogIdsByPath}
                  />
                )}
              </>
            ) : (
              <div className="px-2 py-1.5 rounded-md bg-zinc-800/60 text-zinc-500 text-xs">
                {videoSettingsMessage || 'Loading generation settings...'}
              </div>
            )}
          </>
        )}
        
        {/* Catalog-aware prompt enhancer — video/IC-LoRA (local-generation-only) or image
            (generation/editing, any backend). Runs either the local Gemma text encoder or, if
            available, Gemini's hosted API. */}
        {enhanceAvailableForMode && enhanceProviderAvailable && (
          <>
            {(canUndoPrompt || canRedoPrompt) && (
              <>
                <button
                  type="button"
                  onClick={onUndoPrompt}
                  disabled={isEnhancingPrompt || !canUndoPrompt}
                  title="Undo (previous prompt)"
                  className="flex items-center justify-center h-7 w-7 rounded-md text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 flex-shrink-0 disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  <Undo2 className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  onClick={onRedoPrompt}
                  disabled={isEnhancingPrompt || !canRedoPrompt}
                  title="Redo (next prompt)"
                  className="flex items-center justify-center h-7 w-7 rounded-md text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 flex-shrink-0 disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  <Redo2 className="h-3.5 w-3.5" />
                </button>
              </>
            )}
            <div className="flex items-center flex-shrink-0">
              <button
                type="button"
                onClick={onEnhancePrompt}
                disabled={!canEnhancePrompt || isEnhancingPrompt}
                title={enhancePromptError ?? 'Enhance prompt'}
                className={`flex items-center gap-1 px-2 py-1.5 text-xs font-medium ${
                  onEnhanceProviderChange ? 'rounded-l-md' : 'rounded-md'
                } ${
                  !canEnhancePrompt || isEnhancingPrompt
                    ? 'text-zinc-600 cursor-not-allowed'
                    : enhancePromptError
                      ? 'text-red-400 hover:bg-red-950/40'
                      : 'text-zinc-300 hover:bg-zinc-800'
                }`}
              >
                {isEnhancingPrompt ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Sparkle className="h-3.5 w-3.5" />
                )}
                {isEnhancingPrompt ? 'Enhancing...' : enhanceProvider === 'api' ? 'Enhance (API)' : 'Enhance'}
              </button>
              {onEnhanceProviderChange && (
                <SettingsDropdown
                  title="ENHANCE VIA"
                  value={enhanceProvider ?? 'local'}
                  onChange={(v) => onEnhanceProviderChange(v === 'api' ? 'api' : 'local')}
                  options={[
                    { value: 'local', label: 'Local (Gemma)' },
                    { value: 'api', label: 'API (Gemini)' },
                  ]}
                  trigger={<ChevronUp className="h-3 w-3 text-zinc-500" />}
                  triggerClassName="rounded-l-none border-l border-zinc-700 px-1"
                />
              )}
            </div>
          </>
        )}

        {/* Generate button */}
        <button
          onClick={onGenerate}
          disabled={isGenerating || !canGenerate || isEnhancingPrompt}
          className={`flex items-center gap-1.5 ml-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all flex-shrink-0 ${
            isGenerating || !canGenerate || isEnhancingPrompt
              ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
              : 'bg-white text-black hover:bg-zinc-200'
          }`}
        >
          <span className={isGenerating ? 'animate-pulse' : ''}>{buttonIcon}</span>
          {buttonLabel}
        </button>
      </div>
    </div>
  )
}

// Gallery size icon components
function GridSmallIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <rect x="2" y="2" width="4" height="4" rx="0.5" />
      <rect x="8" y="2" width="4" height="4" rx="0.5" />
      <rect x="14" y="2" width="4" height="4" rx="0.5" />
      <rect x="20" y="2" width="2" height="4" rx="0.5" />
      <rect x="2" y="8" width="4" height="4" rx="0.5" />
      <rect x="8" y="8" width="4" height="4" rx="0.5" />
      <rect x="14" y="8" width="4" height="4" rx="0.5" />
      <rect x="20" y="8" width="2" height="4" rx="0.5" />
      <rect x="2" y="14" width="4" height="4" rx="0.5" />
      <rect x="8" y="14" width="4" height="4" rx="0.5" />
      <rect x="14" y="14" width="4" height="4" rx="0.5" />
      <rect x="20" y="14" width="2" height="4" rx="0.5" />
    </svg>
  )
}

function GridMediumIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <rect x="2" y="2" width="6" height="6" rx="1" />
      <rect x="10" y="2" width="6" height="6" rx="1" />
      <rect x="18" y="2" width="4" height="6" rx="1" />
      <rect x="2" y="10" width="6" height="6" rx="1" />
      <rect x="10" y="10" width="6" height="6" rx="1" />
      <rect x="18" y="10" width="4" height="6" rx="1" />
      <rect x="2" y="18" width="6" height="4" rx="1" />
      <rect x="10" y="18" width="6" height="4" rx="1" />
      <rect x="18" y="18" width="4" height="4" rx="1" />
    </svg>
  )
}

function GridLargeIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <rect x="2" y="2" width="9" height="9" rx="1.5" />
      <rect x="13" y="2" width="9" height="9" rx="1.5" />
      <rect x="2" y="13" width="9" height="9" rx="1.5" />
      <rect x="13" y="13" width="9" height="9" rx="1.5" />
    </svg>
  )
}

type GallerySize = 'small' | 'medium' | 'large'

const gallerySizeClasses: Record<GallerySize, string> = {
  small: 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-7',
  medium: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5',
  large: 'grid-cols-1 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-3',
}

const DEFAULT_VIDEO_SETTINGS = {
  model: 'fast',
  duration: 5,
  videoResolution: '540p',
  fps: 24,
  aspectRatio: '16:9',
  imageResolution: '1080p',
  variations: 1,
  audio: true,
  imageEditStrength: 0.6,
}

export function GenSpace() {
  const {
    activeProject,
    addAsset,
    addTakeToAsset,
    deleteAsset,
    toggleFavorite,
    genSpaceEditImagePath,
    setGenSpaceEditImagePath,
    setGenSpaceEditMode,
    genSpaceAudioPath,
    setGenSpaceAudioPath,
    genSpaceRetakeSource,
    setGenSpaceRetakeSource,
    setPendingRetakeUpdate,
    genSpaceIcLoraSource,
    setGenSpaceIcLoraSource,
    setPendingIcLoraUpdate,
  } = useProjects()
  const currentProjectId = activeProject?.id ?? null
  const { shouldVideoGenerateWithLtxApi, shouldImageGenerateWithFalApi, forceApiGenerations, settings: appSettings } = useAppSettings()
  const {
    modelSpecs: videoGenerationModelSpecsResponse,
    isLoading: isLoadingVideoGenerationModelSpecs,
    errorMessage: videoGenerationModelSpecsErrorMessage,
  } = useVideoGenerationModelSpecs()
  const [mode, setMode] = useState<GenSpaceMode>('video')
  const [prompt, setPrompt] = useState('')
  const [isEnhancingPrompt, setIsEnhancingPrompt] = useState(false)
  const [enhancePromptError, setEnhancePromptError] = useState<string | null>(null)
  // Undo/redo stack for the prompt enhancer, e.g. [original, enhance#1, enhance#2].
  // historyIndex points at whichever entry is currently shown in the textarea. Undo/Redo are
  // pure local navigation (no network call) — only a fresh Enhance call ever grows the stack.
  const [promptHistory, setPromptHistory] = useState<string[]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)
  const [inputImage, setInputImage] = useState<string | null>(null)
  const [inputAudio, setInputAudio] = useState<string | null>(null)
  const [localError, setLocalError] = useState<GenerationError | null>(null)
  const [selectedAsset, setSelectedAsset] = useState<Asset | null>(null)
  const [copiedPrompt, setCopiedPrompt] = useState(false)
  const [showFavorites, setShowFavorites] = useState(false)
  const [gallerySize, setGallerySize] = useState<GallerySize>('medium')
  const [showSizeMenu, setShowSizeMenu] = useState(false)
  const sizeMenuRef = useRef<HTMLDivElement>(null)
  const persistedVideoKeyRef = useRef<string | null>(null)
  const retakeSubmissionRef = useRef<{
    prompt: string
    input: {
      videoPath: string | null
      startTime: number
      duration: number
      videoDuration: number
    }
  } | null>(null)
  const icLoraSubmissionRef = useRef<{
    prompt: string
    input: {
      videoPath: string
      conditioningType: ICLoraConditioningType
      conditioningStrength: number
      customLoraRef?: string
    }
  } | null>(null)
  // Provenance for the completion effect below: imagePaths/isGenerating alone can't tell
  // it whether the request that just finished was an edit or a plain generation.
  const lastImageEditRef = useRef<{ source: string; strength: number } | null>(null)
  const [settings, setSettings] = useState(() => ({ ...DEFAULT_VIDEO_SETTINGS }))
  const videoModelSpecs = getVideoGenerationModelSpecs(videoGenerationModelSpecsResponse, {
    useApiSpecs: shouldVideoGenerateWithLtxApi,
  })
  const videoSettingsMessage = isLoadingVideoGenerationModelSpecs
    ? 'Loading generation settings...'
    : videoGenerationModelSpecsErrorMessage
      ? `Could not load generation settings: ${videoGenerationModelSpecsErrorMessage}`
      : null
  const sanitizeVideoSettings = useCallback(
    (next: typeof settings) => {
      if (mode !== 'video' || videoModelSpecs.length === 0) return next
      return sanitizeVideoGenerationSettings(next, videoModelSpecs, {
        hasAudio: Boolean(inputAudio),
      }) ?? next
    },
    [inputAudio, mode, videoModelSpecs],
  )
  
  const {
    generate,
    generateImage,
    isGenerating,
    progress,
    statusMessage,
    videoPath,
    imagePaths,
    error,
    reset,
    resumeIfRunning,
  } = useGeneration()

  // Locally installed LoRAs are only usable in local generation mode.
  const isLocalMode = !shouldVideoGenerateWithLtxApi
  // Enhance itself is independent of the video-generation backend — the backend enhance
  // endpoint only cares about the enhancer provider (local Gemma vs. Gemini), not whether video
  // generation runs locally or via the LTX API. If no catalog LoRA is selected (e.g. because the
  // LoRA picker is local-only), it just falls back to a generic rewrite. "retake"/"extend" have
  // no prompt input, so they're excluded.
  const enhanceAvailableForMode = mode === 'video' || mode === 'ic-lora' || mode === 'image'
  // The prompt enhancer can run the local Gemma text encoder OR Gemini's hosted API — this hook
  // tracks which of those is actually available (not just which the user prefers) and picks
  // whichever provider Enhance should use. Refetched whenever the user is in a mode the button
  // could appear in, so downloading the checkpoint from Settings and coming back here picks it up.
  const {
    isAvailable: isEnhancerProviderAvailable,
    provider: enhanceProvider,
    canToggleProvider: canToggleEnhanceProvider,
    setProviderPreference: setEnhanceProviderPref,
  } = usePromptEnhancerProvider(enhanceAvailableForMode)
  const isCustomIcLoraEnabled = useCustomIcLoraEnabled()
  const [localLoras, setLocalLoras] = useState<ApiSuccessOf<'listModels'>['models']>([])
  const [selectedLoras, setSelectedLoras] = useState<LoraSelection[]>([])
  // Installed IC-LoRAs (detected by metadata) for the "Custom IC-LoRA" picker.
  const [installedIcLoras, setInstalledIcLoras] = useState<ApiSuccessOf<'listModels'>['models']>([])
  const [icLoraCustomRef, setIcLoraCustomRef] = useState<string | null>(null)
  const [icLoraSkipStage2, setIcLoraSkipStage2] = useState(false)
  const [icLoraUseLoraInStage2, setIcLoraUseLoraInStage2] = useState(false)
  const [icLoraResolutionFactor, setIcLoraResolutionFactor] = useState(2.0)
  const [icLoraAudioMode, setIcLoraAudioMode] = useState<IcLoraAudioMode>('generated')
  const [icLoraLoraStrength, setIcLoraLoraStrength] = useState(1.0)
  const [icLoraFps, setIcLoraFps] = useState<number | null>(null)

  // IC-LoRA mode: a downloadable curated IC-LoRA that resolves its own weights and builds
  // the control video server-side. When an IC-LoRA is selected, it supersedes canny/depth/custom.
  const { advancedIcLoraControls } = useDevFlags().flags
  // Values for the selected IC-LoRA's catalog controls, keyed by control id. Seeded from each
  // control's default on selection; the settings row renders + edits them generically.
  const [controlValues, setControlValues] = useState<Record<string, number | string>>({})
  // Outpainting per-edge pads (the position_canvas control's structured value, sent via outpaint_pads).
  const [outpaintPads, setOutpaintPads] = useState<OutpaintPads>({ left: 0, right: 0, top: 0, bottom: 0 })
  // Selecting an IC-LoRA seeds the knob state from its default_settings so the advanced
  // controls (when shown) reflect what the IC-LoRA will actually run with. (useIcLoraLibrary owns
  // the selected id; this only seeds the generation state it can't reach.)
  const onSelectIcLora = useCallback((item: IcLoraListItem | null) => {
    // Seed option-based controls from their defaults; position_canvas has no default (value is pads).
    setControlValues(Object.fromEntries(
      (item?.ic_lora.controls ?? [])
        .filter(c => c.default !== null)
        .map(c => [c.id, c.default as number | string]),
    ))
    setOutpaintPads({ left: 0, right: 0, top: 0, bottom: 0 })
    const s = item?.ic_lora.default_settings
    if (s) {
      setIcLoraSkipStage2(s.skip_stage_2 ?? false)
      setIcLoraUseLoraInStage2(s.use_lora_in_stage_2 ?? false)
      setIcLoraResolutionFactor(s.resolution_factor ?? 2.0)
      setIcLoraAudioMode(s.audio_mode ?? 'generated')
      setIcLoraLoraStrength(s.lora_strength ?? 1.0)
      setIcLoraStrength(s.conditioning_strength ?? 1.0)
    }
  }, [])
  const refreshInstalledModels = useCallback(() => {
    if (shouldVideoGenerateWithLtxApi) {
      setLocalLoras([])
      setInstalledIcLoras([])
      return
    }
    ApiClient.listModels({ type: 'lora' }).then(result => {
      if (result.ok) setLocalLoras(result.data.models)
    })
    ApiClient.listModels({ type: 'ic-lora' }).then(result => {
      if (result.ok) setInstalledIcLoras(result.data.models)
    })
  }, [shouldVideoGenerateWithLtxApi])
  useEffect(() => { refreshInstalledModels() }, [refreshInstalledModels])

  const {
    icLoras, items: icLoraItems, downloadIcLora, downloadingKey: downloadingIcLoraKey,
    progress: icLoraDownloadProgress, downloadError: icLoraDownloadError,
    modalOpen: libraryModalOpen, setModalOpen: setLibraryModalOpen,
    selectedIcLoraId, selectedIcLoraVariantId, selectIcLora,
  } = useIcLoraLibrary(
    mode === 'ic-lora' && !shouldVideoGenerateWithLtxApi,
    onSelectIcLora,
    installedIcLoras,
    refreshInstalledModels,
  )
  const selectedIcLora = icLoras.find(r => r.ic_lora.id === selectedIcLoraId)?.ic_lora ?? null
  const isCatalogIcLora = selectedIcLora !== null
  const hasPositionCanvas = (selectedIcLora?.controls ?? []).some(c => c.kind === 'position_canvas')
  // Catalog IC-LoRAs may opt into promptless generation (e.g. outpainting fills from the scene).
  const promptOptional = isCatalogIcLora && (selectedIcLora?.allows_empty_prompt ?? false)
  const onControlChange = useCallback((id: string, value: number | string) => {
    setControlValues(prev => ({ ...prev, [id]: value }))
  }, [])

  // Plain-LoRA library: catalog browse/download + on-disk merge + "use" → selectedLoras.
  const loraLibrary = useLoraLibrary(isLocalMode, localLoras, selectedLoras, setSelectedLoras, refreshInstalledModels)
  // installed path -> catalog display name (from the catalog↔on-disk merge), so the inline
  // LoRA picker shows a downloaded catalog LoRA's proper name instead of its raw filename.
  const loraDisplayNames = useMemo(() => {
    const map = new Map<string, string>()
    for (const i of loraLibrary.items) {
      if (i.variantInstalledPaths && i.variants) {
        for (const v of i.variants) {
          const path = i.variantInstalledPaths[v.id]
          if (path) map.set(path, variantDisplayName(i.name, v.label, i.variants.length))
        }
      } else if (i.installedPath) {
        map.set(i.installedPath, i.name)
      }
    }
    return map
  }, [loraLibrary.items])

  // Catalog id keyed by installed path — same cross-check as loraDisplayNames, so the LoRAPicker
  // dropdown (list of already-installed files) can attach catalogId too, not just the "Browse
  // LoRAs" modal flow. Without this, picking a catalog LoRA from this dropdown instead of the
  // modal silently loses catalog awareness for the prompt enhancer.
  const loraCatalogIdsByPath = useMemo(() => {
    const map = new Map<string, string>()
    for (const i of loraLibrary.items) {
      if (i.variantInstalledPaths) {
        for (const path of Object.values(i.variantInstalledPaths)) {
          if (path) map.set(path, i.id)
        }
      } else if (i.installedPath) {
        map.set(i.installedPath, i.id)
      }
    }
    return map
  }, [loraLibrary.items])

  const {
    submitRetake,
    resetRetake,
    isRetaking,
    retakeStatus,
    retakeError,
    retakeResult,
  } = useRetake()

  const [retakeInput, setRetakeInput] = useState({
    videoPath: null as string | null,
    startTime: 0,
    duration: 0,
    videoDuration: 0,
    width: 0,
    height: 0,
    ready: false,
  })
  const [retakeResolutionKey, setRetakeResolutionKey] = useState('original')
  const [retakePanelKey, setRetakePanelKey] = useState(0)

  const {
    submitExtend,
    resetExtend,
    isExtending,
    extendStatus,
    extendError,
    extendResult,
  } = useExtend()
  const [extendInput, setExtendInput] = useState({
    videoPath: null as string | null,
    videoDuration: 0,
    width: 0,
    height: 0,
    ready: false,
  })
  // Direction + seconds + resolution live here (controlled by the PromptBar), not the panel.
  const [extendDirection, setExtendDirection] = useState<ExtendDirection>('end')
  const [extendSeconds, setExtendSeconds] = useState<number>(DEFAULT_EXTEND_SECONDS)
  const [extendResolutionKey, setExtendResolutionKey] = useState('original')

  const retakeResolutionOpts = useMemo(
    () => resolutionOptions(retakeInput.width, retakeInput.height),
    [retakeInput.width, retakeInput.height],
  )
  const extendResolutionOpts = useMemo(
    () => resolutionOptions(extendInput.width, extendInput.height),
    [extendInput.width, extendInput.height],
  )
  const [extendPanelKey, setExtendPanelKey] = useState(0)
  const [extendInitial, setExtendInitial] = useState<{
    videoPath: string | null
    duration?: number
  }>({ videoPath: null, duration: undefined })
  const extendSubmissionRef = useRef<{
    prompt: string
    input: { videoPath: string; direction: ExtendDirection; duration: number; videoDuration: number }
  } | null>(null)
  const [retakeInitial, setRetakeInitial] = useState<{
    videoPath: string | null
    duration?: number
  }>({ videoPath: null, duration: undefined })
  const [activeRetakeSource, setActiveRetakeSource] = useState<GenSpaceRetakeSource | null>(null)
  const [activeIcLoraSource, setActiveIcLoraSource] = useState<{
    assetId?: string
    linkedClipIds?: string[]
  } | null>(null)
  const [icLoraInput, setIcLoraInput] = useState({
    videoPath: null as string | null,
    conditioningType: 'canny' as ICLoraConditioningType,
    conditioningStrength: 1.0,
    ready: false,
    referenceImagePath: null as string | null,
    width: 0,
    height: 0,
  })
  // Resolution tiers for the use_lora_in_stage_2 two-stage path (mirrors retake/extend).
  const [icLoraResolutionKey, setIcLoraResolutionKey] = useState('original')
  const icLoraResolutionOpts = useMemo(
    () => resolutionOptions(icLoraInput.width, icLoraInput.height),
    [icLoraInput.width, icLoraInput.height],
  )
  const [icLoraPanelKey, setIcLoraPanelKey] = useState(0)
  const [icLoraCondType, setIcLoraCondType] = useState<ICLoraConditioningType>('canny')
  // Transformation knobs apply only to custom IC-LoRAs. Reset them when switching to
  // canny/depth (control LoRAs) so those never deviate from the original two-stage behavior.
  const handleIcLoraCondTypeChange = (type: ICLoraConditioningType) => {
    setIcLoraCondType(type)
    // Picking a conditioning type leaves catalog IC-LoRA mode (the dropdown is the single writer).
    selectIcLora(null)
    if (type !== 'custom') {
      setIcLoraSkipStage2(false)
      setIcLoraUseLoraInStage2(false)
      setIcLoraResolutionKey('original')
      setIcLoraResolutionFactor(2.0)
      setIcLoraFps(null)
      setIcLoraAudioMode('generated')
      setIcLoraLoraStrength(1.0)
    }
  }
  // Unified IC-LoRA selector: one dropdown drives canny/depth, downloaded catalog
  // IC-LoRAs (one option per installed variant) and custom. selectedIcLoraId +
  // selectedIcLoraVariantId + icLoraCondType remain the underlying state.
  const icLoraSelectorOptions = buildIcLoraSelectorOptions(icLoraItems, CONDITIONING_TYPES, {
    includeCatalog: true,
    includeCustom: isCustomIcLoraEnabled,
  })
  const selectedIcLoraEntry = icLoraItems.find(e => e.id === selectedIcLoraId)
  const icLoraSelectorValue = selectedIcLoraId
    ? encodeIcLoraSelectorValue(selectedIcLoraId, selectedIcLoraVariantId, {
      variants: selectedIcLoraEntry?.variants,
      downloadedVariantIds: selectedIcLoraEntry?.downloadedVariantIds,
    })
    : (icLoraCondType || 'canny')
  const handleIcLoraSelectorChange = (value: string) => {
    const { catalogId, variantId } = parseIcLoraSelectorValue(value)
    const item = icLoras.find(r => r.ic_lora.id === catalogId)
    if (item) selectIcLora(item, variantId)
    else handleIcLoraCondTypeChange(value as ICLoraConditioningType)
  }
  const [icLoraStrength, setIcLoraStrength] = useState(1.0)
  const [icLoraInitial, setIcLoraInitial] = useState<{
    videoPath: string | null
  }>({ videoPath: null })

  const {
    submitIcLora,
    resetIcLora,
    isIcLoraGenerating,
    icLoraStatus,
    icLoraError,
    icLoraResult,
  } = useIcLora()
  
  // Handle incoming frame from the Video Editor for editing
  useEffect(() => {
    if (genSpaceEditImagePath) {
      setMode('video')
      setInputImage(genSpaceEditImagePath)
      setPrompt('')
      setGenSpaceEditImagePath(null)
      setGenSpaceEditMode(null)
    }
  }, [genSpaceEditImagePath, setGenSpaceEditImagePath, setGenSpaceEditMode])

  // Handle incoming audio from the Video Editor for A2V
  useEffect(() => {
    if (genSpaceAudioPath) {
      setMode('video')
      setInputAudio(genSpaceAudioPath)
      setPrompt('')
      setGenSpaceAudioPath(null)
    }
  }, [genSpaceAudioPath, setGenSpaceAudioPath])

  useEffect(() => {
    if (!genSpaceRetakeSource) return
    setMode('retake')
    setPrompt('')
    setActiveRetakeSource(genSpaceRetakeSource)
    setRetakeInitial({
      videoPath: genSpaceRetakeSource.videoPath,
      duration: genSpaceRetakeSource.duration,
    })
    setRetakePanelKey((prev) => prev + 1)
    setGenSpaceRetakeSource(null)
  }, [genSpaceRetakeSource, setGenSpaceRetakeSource])

  useEffect(() => {
    if (!genSpaceIcLoraSource) return
    if (forceApiGenerations) {
      setGenSpaceIcLoraSource(null)
      return
    }
    setMode('ic-lora')
    setPrompt('')
    setActiveIcLoraSource({
      assetId: genSpaceIcLoraSource.assetId,
      linkedClipIds: genSpaceIcLoraSource.linkedClipIds,
    })
    setIcLoraInitial({
      videoPath: genSpaceIcLoraSource.videoPath,
    })
    setIcLoraCustomRef(null)
    setIcLoraPanelKey((prev) => prev + 1)
    setGenSpaceIcLoraSource(null)
  }, [genSpaceIcLoraSource, forceApiGenerations, setGenSpaceIcLoraSource])

  useEffect(() => {
    if (forceApiGenerations && mode === 'ic-lora') {
      setMode('video')
    }
  }, [forceApiGenerations, mode])

  useEffect(() => {
    if (mode !== 'video' || videoModelSpecs.length === 0) return
    setSettings((prev) => {
      const next = sanitizeVideoSettings(prev)
      return areVideoGenerationSettingsEquivalent(prev, next) ? prev : next
    })
  }, [mode, sanitizeVideoSettings, videoModelSpecs.length])

  useEffect(() => {
    if (retakeError) {
      setLocalError(createLocalGenerationError(retakeError))
    }
  }, [retakeError])

  useEffect(() => {
    if (extendError) {
      setLocalError(createLocalGenerationError(extendError))
    }
  }, [extendError])

  useEffect(() => {
    if (icLoraError) {
      setLocalError(createLocalGenerationError(icLoraError))
    }
  }, [icLoraError])

  // Only show assets that were generated (have generationParams), not imported files
  const assets = (activeProject?.assets || []).filter(a => a.generationParams)
  const [lastPrompt, setLastPrompt] = useState('')

  // On mount: recover any generation that was still running when the frontend reloaded.
  const currentProjectIdRef = useRef(currentProjectId)
  currentProjectIdRef.current = currentProjectId

  // Tell the background recovery watcher (App.tsx) that THIS mount has a live generation request
  // of its own in flight for this project — it backs off for this project id so the two never
  // race to import the same completion twice. Deliberately scoped to "actually generating", not
  // just "mounted for this project": a mount that's only here to recover a marker left by an
  // earlier mount (e.g. the user left mid-generation and came back before it finished) has
  // nothing live of its own, and must NOT block the watcher — otherwise nothing ever revisits
  // that marker again for as long as this mount stays open, even long after the generation
  // backing it actually finishes. Also covers the "isGenerating just flipped false, completion
  // effect below is still copying the result into project storage" window — the marker is only
  // removed once that effect's own reset()/resetX() runs, so ownership must last at least that
  // long too, or the watcher can import the same completion a second time.
  const isAnyLocalGenerationInFlight = isGenerating || isRetaking || isExtending || isIcLoraGenerating
    || !!videoPath || imagePaths.length > 0 || !!retakeResult || !!extendResult || !!icLoraResult
  useEffect(() => {
    if (!isAnyLocalGenerationInFlight) return
    setActiveGenerationOwner(currentProjectId)
    return () => setActiveGenerationOwner(null)
  }, [currentProjectId, isAnyLocalGenerationInFlight])

  useEffect(() => {
    const saved = localStorage.getItem(GENERATION_RECOVERY_KEY)
    if (!saved) return
    let ctx: GenerationRecoveryContext
    try { ctx = JSON.parse(saved) as GenerationRecoveryContext } catch { return }
    if (!hasValidBaselineId(ctx)) return // legacy/corrupt marker — the watcher will drop it
    // Belongs to a different project (e.g. the user started this generation in project A, then
    // navigated Home and into project B — GenSpace remounts per project). Not ours to touch.
    if (ctx.projectId !== currentProjectIdRef.current) return
    // Enhance recovery is handled by its own effect (different state: isEnhancingPrompt, not
    // isGenerating/videoPath/imagePath) — leave the marker for it to consume.
    if (ctx.genType === 'enhance') return

    void (async () => {
      const progress = await ApiClient.getGenerationProgress()
      if (!progress.ok) return
      // Same identity check as the background watcher (checkAndConsumeRecovery in
      // lib/generation-recovery.ts): the handler that starts a generation loads its pipeline —
      // which can take many seconds, worse for image models loading checkpoint shards — BEFORE
      // it ever reports a new id, so a poll right after remounting can still be looking at
      // whatever the single global progress slot held before this marker was even written.
      if (progress.data.id === ctx.baselineId) return // unchanged since before we wrote this marker — not started reporting yet
      if (progress.data.status !== 'running') return // already past 'running' (complete/error/etc) — that's the watcher's job to persist, not ours to restore UI for

      const status = await resumeIfRunning()
      if (status !== 'running') return // finished between our two checks — again, the watcher's job now

      // Restore the inputs the completion effect reads, so the recovered asset is eventually
      // persisted (once it finishes) with the right prompt/mode/settings (incl. selected LoRAs),
      // not an empty prompt and default settings.
      setPrompt(ctx.prompt)
      setLastPrompt(ctx.prompt)
      // ic-lora/retake carry no settings: recover as a standalone video asset.
      const s = ctx.settings
      if (!s) {
        setMode('video')
      } else {
        setMode(ctx.genType === 'image' ? 'image' : 'video')
        setInputImage(ctx.inputImageUrl ?? null)
        setInputAudio(ctx.inputAudioUrl ?? null)
        setSelectedLoras(s.loras ?? [])
        setSettings(prev => ({
          ...prev,
          model: s.model,
          duration: s.duration,
          videoResolution: s.videoResolution,
          fps: s.fps,
          audio: s.audio,
          aspectRatio: s.aspectRatio ?? prev.aspectRatio,
          imageResolution: s.imageResolution,
          variations: s.variations ?? prev.variations,
          imageEditStrength: s.imageEditStrength ?? prev.imageEditStrength,
        }))
        // The completion effect reads this ref (not settings/inputImage) to tag a
        // recovered image asset as an edit — restore it so recovery matches the
        // live handleGenerate() path.
        if (ctx.genType === 'image' && ctx.inputImageUrl) {
          lastImageEditRef.current = { source: ctx.inputImageUrl, strength: s.imageEditStrength ?? 0.6 }
        }
      }
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // mount only

  // When video generation completes, add to project assets
  useEffect(() => {
    if (!videoPath || !currentProjectId || isGenerating) return

    // Dedup is by persistedVideoKeyRef, not an assets path-match like the image effect:
    // addVisualAssetToProject copies to a fresh path so the source videoPath never appears
    // in assets. A reload can't double-import either — recovery clears the marker once it
    // reaches 'complete' (use-generation), so the sticky output is only restored once.
    const generationKey = videoPath
    if (persistedVideoKeyRef.current === generationKey) return
    persistedVideoKeyRef.current = generationKey

    const genMode = inputAudio
      ? 'audio-to-video'
      : inputImage ? 'image-to-video' : 'text-to-video'
    const savedVideoSettings = sanitizeVideoSettings(settings)

    ;(async () => {
      try {
        const copied = await addVisualAssetToProject(videoPath, currentProjectId, 'video')
        if (!copied) throw new Error('Could not persist generated video to project storage')
        addAsset(currentProjectId, {
          type: 'video',
          path: copied.path,
          bigThumbnailPath: copied.bigThumbnailPath,
          smallThumbnailPath: copied.smallThumbnailPath,
          width: copied.width,
          height: copied.height,
          prompt: lastPrompt,
          resolution: savedVideoSettings.videoResolution,
          duration: savedVideoSettings.duration,
          generationParams: {
            mode: genMode as 'text-to-video' | 'image-to-video' | 'audio-to-video',
            prompt: lastPrompt,
            model: savedVideoSettings.model,
            duration: savedVideoSettings.duration,
            resolution: savedVideoSettings.videoResolution,
            fps: savedVideoSettings.fps,
            audio: savedVideoSettings.audio || false,
            cameraMotion: 'none',
            imageAspectRatio: savedVideoSettings.aspectRatio,
            imageSteps: 4,
            inputImageUrl: inputImage || undefined,
            inputAudioUrl: inputAudio || undefined,
            loras: isLocalMode && selectedLoras.length > 0
              ? selectedLoras.map(l => ({ ...l, ref: toModelsDirRelativeRef(l.ref, appSettings.modelsDir) }))
              : undefined,
          },
          takes: [{
            path: copied.path,
            bigThumbnailPath: copied.bigThumbnailPath,
            smallThumbnailPath: copied.smallThumbnailPath,
            width: copied.width,
            height: copied.height,
            createdAt: Date.now(),
          }],
          activeTakeIndex: 0,
        })
        reset()
      } catch (err) {
        persistedVideoKeyRef.current = null
        logger.error(`Failed to persist generated video asset: ${err}`)
      }
    })()
  }, [videoPath, currentProjectId, isGenerating, sanitizeVideoSettings, settings, inputImage, inputAudio, lastPrompt, addAsset, reset, selectedLoras, isLocalMode, appSettings.modelsDir])

  // When retake completes, add as take or new asset
  useEffect(() => {
    if (!retakeResult || !currentProjectId || isRetaking) return
    const submission = retakeSubmissionRef.current
    if (!submission) return
    retakeSubmissionRef.current = null
    // Imported in-page; drop the recovery marker so a remount can't re-import the
    // backend's sticky GenerationComplete as a duplicate standalone asset.
    localStorage.removeItem(GENERATION_RECOVERY_KEY)

    ;(async () => {
      const usedPrompt = submission.prompt
      const usedInput = submission.input
      const copied = await addVisualAssetToProject(retakeResult.videoPath, currentProjectId, 'video')
      if (!copied) {
        logger.error('Could not persist retake result to project storage')
        setLocalError(createLocalGenerationError('Failed to save retake output to project storage.'))
        setActiveRetakeSource(null)
        resetRetake()
        return
      }

      if (activeRetakeSource?.assetId) {
        const sourceAsset = activeProject?.assets?.find(a => a.id === activeRetakeSource.assetId)
        if (sourceAsset) {
          const newTakeIndex = sourceAsset.takes ? sourceAsset.takes.length : 1
          addTakeToAsset(currentProjectId, sourceAsset.id, {
            path: copied.path,
            bigThumbnailPath: copied.bigThumbnailPath,
            smallThumbnailPath: copied.smallThumbnailPath,
            width: copied.width,
            height: copied.height,
            createdAt: Date.now(),
          })
          if (activeRetakeSource.linkedClipIds?.length) {
            setPendingRetakeUpdate({
              assetId: sourceAsset.id,
              clipIds: activeRetakeSource.linkedClipIds,
              newTakeIndex,
            })
          }
        }
      } else {
        addAsset(currentProjectId, {
          type: 'video',
          path: copied.path,
          bigThumbnailPath: copied.bigThumbnailPath,
          smallThumbnailPath: copied.smallThumbnailPath,
          width: copied.width,
          height: copied.height,
          prompt: usedPrompt,
          resolution: '',
          duration: usedInput.duration,
          generationParams: {
            mode: 'retake',
            prompt: usedPrompt,
            model: 'pro',
            duration: usedInput.duration,
            resolution: '',
            fps: 24,
            audio: true,
            cameraMotion: 'none',
            retakeVideoPath: copied.path,
            retakeStartTime: usedInput.startTime,
            retakeDuration: usedInput.duration,
            retakeMode: 'replace_audio_and_video',
          },
          takes: [{
            path: copied.path,
            bigThumbnailPath: copied.bigThumbnailPath,
            smallThumbnailPath: copied.smallThumbnailPath,
            width: copied.width,
            height: copied.height,
            createdAt: Date.now(),
          }],
          activeTakeIndex: 0,
        })
        setMode('video')
      }

      setActiveRetakeSource(null)
      resetRetake()
    })()
  }, [retakeResult, isRetaking, currentProjectId, activeProject?.assets, activeRetakeSource, addAsset, addTakeToAsset, setPendingRetakeUpdate, resetRetake])

  // When extend completes, save the longer video as a new asset.
  useEffect(() => {
    if (!extendResult || !currentProjectId || isExtending) return
    const submission = extendSubmissionRef.current
    if (!submission) return
    extendSubmissionRef.current = null
    localStorage.removeItem(GENERATION_RECOVERY_KEY)

    ;(async () => {
      const usedPrompt = submission.prompt
      const usedInput = submission.input
      const copied = await addVisualAssetToProject(extendResult.videoPath, currentProjectId, 'video')
      if (!copied) {
        logger.error('Could not persist extend result to project storage')
        setLocalError(createLocalGenerationError('Failed to save extend output to project storage.'))
        resetExtend()
        return
      }

      addAsset(currentProjectId, {
        type: 'video',
        path: copied.path,
        bigThumbnailPath: copied.bigThumbnailPath,
        smallThumbnailPath: copied.smallThumbnailPath,
        width: copied.width,
        height: copied.height,
        prompt: usedPrompt,
        resolution: '',
        duration: usedInput.videoDuration + usedInput.duration,
        generationParams: {
          mode: 'extend',
          prompt: usedPrompt,
          model: 'pro',
          duration: usedInput.videoDuration + usedInput.duration,
          resolution: '',
          fps: 24,
          audio: true,
          cameraMotion: 'none',
          extendVideoPath: copied.path,
          extendDuration: usedInput.duration,
          extendDirection: usedInput.direction,
        },
        takes: [{
          path: copied.path,
          bigThumbnailPath: copied.bigThumbnailPath,
          smallThumbnailPath: copied.smallThumbnailPath,
          width: copied.width,
          height: copied.height,
          createdAt: Date.now(),
        }],
        activeTakeIndex: 0,
      })
      setMode('video')
      resetExtend()
    })()
  }, [extendResult, isExtending, currentProjectId, addAsset, resetExtend])

  useEffect(() => {
    if (!icLoraResult || !currentProjectId || isIcLoraGenerating) return
    const submission = icLoraSubmissionRef.current
    if (!submission) return
    icLoraSubmissionRef.current = null
    // Imported in-page; drop the recovery marker so a remount can't re-import the
    // backend's sticky GenerationComplete as a duplicate standalone asset.
    localStorage.removeItem(GENERATION_RECOVERY_KEY)

    ;(async () => {
      const copied = await addVisualAssetToProject(icLoraResult.videoPath, currentProjectId, 'video')
      if (!copied) {
        logger.error('Could not persist IC-LoRA result to project storage')
        setLocalError(createLocalGenerationError('Failed to save IC-LoRA output to project storage.'))
        setActiveIcLoraSource(null)
        resetIcLora()
        return
      }

      if (activeIcLoraSource?.assetId) {
        const sourceAsset = activeProject?.assets?.find(a => a.id === activeIcLoraSource.assetId)
        if (sourceAsset) {
          const newTakeIndex = sourceAsset.takes ? sourceAsset.takes.length : 1
          addTakeToAsset(currentProjectId, sourceAsset.id, {
            path: copied.path,
            bigThumbnailPath: copied.bigThumbnailPath,
            smallThumbnailPath: copied.smallThumbnailPath,
            width: copied.width,
            height: copied.height,
            createdAt: Date.now(),
          })
          if (activeIcLoraSource.linkedClipIds?.length) {
            setPendingIcLoraUpdate({
              assetId: sourceAsset.id,
              clipIds: activeIcLoraSource.linkedClipIds,
              newTakeIndex,
            })
          }
        }
      } else {
        addAsset(currentProjectId, {
          type: 'video',
          path: copied.path,
          bigThumbnailPath: copied.bigThumbnailPath,
          smallThumbnailPath: copied.smallThumbnailPath,
          width: copied.width,
          height: copied.height,
          prompt: submission.prompt,
          resolution: '',
          generationParams: {
            mode: 'ic-lora',
            prompt: submission.prompt,
            model: 'fast',
            duration: 0,
            resolution: '',
            fps: 24,
            audio: false,
            cameraMotion: 'none',
            icLoraVideoPath: submission.input.videoPath,
            icLoraConditioningType: submission.input.conditioningType,
            icLoraConditioningStrength: submission.input.conditioningStrength,
          },
          takes: [{
            path: copied.path,
            bigThumbnailPath: copied.bigThumbnailPath,
            smallThumbnailPath: copied.smallThumbnailPath,
            width: copied.width,
            height: copied.height,
            createdAt: Date.now(),
          }],
          activeTakeIndex: 0,
        })
      }

      setActiveIcLoraSource(null)
    })()
  }, [icLoraResult, isIcLoraGenerating, currentProjectId, activeProject?.assets, activeIcLoraSource, addAsset, addTakeToAsset, setPendingIcLoraUpdate])
  
  // When image generation/editing completes, add all images to project assets.
  // Dedupe on the *source* path we loop over (imagePaths), tracked in a ref: the stored
  // asset uses the copied project-storage path, so an assets.some(a => a.path === imgPath)
  // check never matches and would re-add on every run. `assets`/`addAsset` are kept out of
  // the deps + an in-flight guard prevents a re-entrant run (addAsset mutating assets would
  // otherwise re-fire this effect while the first pass is still awaiting).
  const addingImagesRef = useRef(false)
  const importedImagePathsRef = useRef<Set<string>>(new Set())
  useEffect(() => {
    if (imagePaths.length === 0 || !currentProjectId || isGenerating) return
    if (addingImagesRef.current) return
    addingImagesRef.current = true
    const editContext = lastImageEditRef.current
    const genMode = editContext ? 'image-edit' : 'text-to-image'

    ;(async () => {
      try {
        for (const imgPath of imagePaths) {
          if (importedImagePathsRef.current.has(imgPath)) continue

          const copied = await addVisualAssetToProject(imgPath, currentProjectId, 'image')
          if (!copied) {
            // Skip this image, keep going — a single failed copy must not drop the rest
            // of the batch (and never reaching reset() below would re-fire this effect).
            logger.error(`Could not persist generated image to project storage: ${imgPath}`)
            continue
          }
          importedImagePathsRef.current.add(imgPath)
          addAsset(currentProjectId, {
            type: 'image',
            path: copied.path,
            bigThumbnailPath: copied.bigThumbnailPath,
            smallThumbnailPath: copied.smallThumbnailPath,
            width: copied.width,
            height: copied.height,
            prompt: lastPrompt,
            resolution: settings.imageResolution,
            generationParams: {
              mode: genMode,
              prompt: lastPrompt,
              model: 'fast',
              duration: 5,
              resolution: settings.imageResolution,
              fps: 24,
              audio: false,
              cameraMotion: 'none',
              imageAspectRatio: settings.aspectRatio,
              imageSteps: editContext ? IMAGE_STEPS_EDIT : IMAGE_STEPS_GENERATE,
              ...(editContext ? { inputImageUrl: editContext.source, imageEditStrength: editContext.strength } : {}),
            },
            takes: [{
              path: copied.path,
              bigThumbnailPath: copied.bigThumbnailPath,
              smallThumbnailPath: copied.smallThumbnailPath,
              width: copied.width,
              height: copied.height,
              createdAt: Date.now(),
            }],
            activeTakeIndex: 0,
          })
        }
        reset()
      } catch (err) {
        logger.error(`Failed to persist generated image asset(s): ${err}`)
      } finally {
        addingImagesRef.current = false
      }
    })()
  }, [imagePaths, currentProjectId, isGenerating, addAsset, lastPrompt, settings, reset])
  
  // Single writer for the recovery marker so the per-mode branches below can't drift. Captures
  // whatever generation id is active RIGHT NOW, before this generation starts, as baselineId —
  // see GenerationRecoveryContext for why: the handler that starts a generation loads its
  // pipeline (can take many seconds) before it ever reports a new id, so without a baseline
  // captured up front, a poll can't tell "my generation hasn't started reporting yet" apart from
  // "a stale, unrelated result predates this marker entirely".
  const writeRecoveryContext = async (ctx: Omit<GenerationRecoveryContext, 'projectId' | 'baselineId'>) => {
    if (!currentProjectId) return
    const before = await ApiClient.getGenerationProgress()
    if (!before.ok) {
      // Fail closed: a failed fetch is indistinguishable from a legitimate idle baseline
      // (both would otherwise write baselineId: null), but if a prior generation is still
      // sticky-complete with a real id, that null baseline lets the very first recovery tick
      // mistake the old generation's result for this one's. No marker means this generation
      // just isn't recoverable if the user navigates away mid-flight — safer than misimporting.
      logger.error(`Skipping recovery marker: failed to fetch baseline generation id (${before.error})`)
      return
    }
    const baselineId = before.data.id ?? null
    logger.info(`Writing recovery marker for ${currentProjectId} (genType=${ctx.genType ?? 'video'}, baselineId=${baselineId})`)
    localStorage.setItem(
      GENERATION_RECOVERY_KEY,
      JSON.stringify({ projectId: currentProjectId, baselineId, ...ctx } satisfies GenerationRecoveryContext),
    )
  }

  // Catalog-aware prompt enhancer: rewrites `prompt` in place using either the local Gemma text
  // encoder or, if available, Gemini's hosted API — informed by whichever catalog LoRA(s)/
  // IC-LoRA are currently selected (or a generic rewrite if none are). Regular-LoRA and IC-LoRA
  // selection are mutually exclusive UI surfaces, so at most one of loraCatalogIds/icLoraId is
  // ever sent.
  // Only one generation can run at a time across the whole app — this catches the case where
  // it's a DIFFERENT project's, which this instance's own isGenerating/isRetaking/etc (all local
  // state) can't see. Without it, Enhance/Generate stayed clickable and the request just 409'd.
  const isOtherGenerationRunning = useGlobalGenerationLock()
  const isGenerationInProgressForEnhance = mode === 'ic-lora' ? isIcLoraGenerating : isGenerating
  const canEnhancePrompt = enhanceAvailableForMode && isEnhancerProviderAvailable
    && prompt.trim().length > 0 && !isGenerationInProgressForEnhance && !isOtherGenerationRunning
  const canUndoPrompt = historyIndex > 0
  const canRedoPrompt = historyIndex >= 0 && historyIndex < promptHistory.length - 1

  // Appends [sourcePrompt (if not already the top of the stack), enhancedPrompt] and truncates
  // any redo entries beyond the current position first — the same rule any undo/redo stack
  // uses: taking a new action after an undo discards the redone-away future.
  const applyEnhanceResult = useCallback((sourcePrompt: string, enhancedPrompt: string) => {
    const truncated = promptHistory.slice(0, historyIndex + 1)
    const withSource = truncated.length > 0 && truncated[truncated.length - 1] === sourcePrompt
      ? truncated
      : [...truncated, sourcePrompt]
    const nextHistory = [...withSource, enhancedPrompt]
    setPromptHistory(nextHistory)
    setHistoryIndex(nextHistory.length - 1)
    setPrompt(enhancedPrompt)
  }, [promptHistory, historyIndex])

  const runEnhance = useCallback(async (sourcePrompt: string) => {
    if (isEnhancingPrompt) return
    setIsEnhancingPrompt(true)
    setEnhancePromptError(null)

    // Recovery marker so a reload mid-enhance can reconnect to the still-running backend call
    // (see the mount effect below) instead of silently losing the result.
    await writeRecoveryContext({ prompt: sourcePrompt, genType: 'enhance' })

    const loraCatalogIds = mode === 'video'
      ? (selectedLoras ?? []).map(l => l.catalogId).filter((id): id is string => !!id)
      : []
    // The "bring your own IC-LoRA" custom flow has no catalog entry — canny/depth conditioning
    // still get a dedicated (non-catalog) system prompt; a fully custom LoRA with neither gets
    // the generic fallback, same as no selection at all.
    const conditioningType = mode === 'ic-lora' && !selectedIcLoraId && (icLoraCondType === 'canny' || icLoraCondType === 'depth')
      ? icLoraCondType
      : undefined

    // inputImage is the image-edit/i2v reference and isn't cleared on mode change — only
    // meaningful in image mode or video's i2v; IC-LoRA's own reference is always a driving
    // video (icLoraInput.videoPath), never this, so a leftover inputImage from an earlier
    // image-mode/i2v session must not leak into an IC-LoRA (or plain t2v) enhance call.
    const imagePathForEnhance = mode === 'image' || mode === 'video' ? inputImage ?? undefined : undefined

    // Local-provider Enhance runs the same GIL-holding Gemma text encoder as local video/image
    // generation (see electron/python-backend.ts) — needs the same liveness-kill suppression.
    const result = await withGenerationActive(() => ApiClient.enhancePrompt({
      prompt: sourcePrompt,
      loraCatalogIds,
      icLoraId: mode === 'ic-lora' ? selectedIcLoraId ?? undefined : undefined,
      conditioningType,
      imagePath: imagePathForEnhance,
      provider: enhanceProvider,
      mediaType: mode === 'image' ? 'image' : 'video',
    }))

    setIsEnhancingPrompt(false)
    if (!result.ok) {
      // Don't clear the marker here: this "failure" can just be our own fetch getting cut by a
      // refresh that's already in progress (backendFetch throws, api-client.ts reports it as a
      // synthetic NETWORK_ERROR) — the backend enhance call itself is a plain in-process Python
      // call, unaffected by the client's connection dying, and keeps running for real. Clearing
      // unconditionally here wiped the marker for a generation that was still active, leaving
      // nothing to reconnect to after the reload. A genuine failure still gets cleaned up once
      // the mount-recovery effect below actually confirms a non-running terminal state.
      logger.info(`Enhance request failed (${result.error.code}), leaving recovery marker for reconciliation`)
      setEnhancePromptError(result.error.message)
      return
    }
    logger.info('Enhance request succeeded, clearing recovery marker')
    localStorage.removeItem(GENERATION_RECOVERY_KEY)
    applyEnhanceResult(sourcePrompt, result.data.enhancedPrompt)
  }, [isEnhancingPrompt, mode, selectedLoras, selectedIcLoraId, icLoraCondType, inputImage, enhanceProvider, applyEnhanceResult, writeRecoveryContext])

  const handleEnhancePrompt = useCallback(() => {
    if (!canEnhancePrompt) return
    void runEnhance(prompt)
  }, [canEnhancePrompt, prompt, runEnhance])

  // Pure local navigation through promptHistory — no network call. "Redo" means redo the
  // undo you just did, not "enhance again" (a fresh rewrite is just Enhance after an Undo).
  const handleUndoPrompt = useCallback(() => {
    if (!canUndoPrompt) return
    const newIndex = historyIndex - 1
    setHistoryIndex(newIndex)
    setPrompt(promptHistory[newIndex])
  }, [canUndoPrompt, historyIndex, promptHistory])

  const handleRedoPrompt = useCallback(() => {
    if (!canRedoPrompt) return
    const newIndex = historyIndex + 1
    setHistoryIndex(newIndex)
    setPrompt(promptHistory[newIndex])
  }, [canRedoPrompt, historyIndex, promptHistory])

  // On mount: reconnect to an enhance that was still running when the frontend reloaded.
  // Separate from the video/image recovery effect above (different state: isEnhancingPrompt,
  // not isGenerating/videoPath/imagePath) — that effect already skips genType === 'enhance'
  // and leaves the marker for this one.
  useEffect(() => {
    const saved = localStorage.getItem(GENERATION_RECOVERY_KEY)
    if (!saved) return
    let ctx: GenerationRecoveryContext
    try { ctx = JSON.parse(saved) as GenerationRecoveryContext } catch { return }
    if (ctx.genType !== 'enhance') return
    if (!hasValidBaselineId(ctx)) {
      logger.warn(`Enhance recovery marker for ${ctx.projectId} has an invalid baselineId — leaving as-is`)
      return // legacy/corrupt marker — nothing to recover
    }
    // Same project-mismatch case as the generation-recovery effect above: belongs to a
    // different (still open elsewhere) project, not stale — leave it for that project's mount.
    if (ctx.projectId !== currentProjectIdRef.current) {
      logger.info(`Enhance recovery marker belongs to ${ctx.projectId}, this mount is ${currentProjectIdRef.current} — leaving it`)
      return
    }

    logger.info(`Reconnecting to in-flight enhance for ${ctx.projectId} (baselineId=${ctx.baselineId})`)
    setIsEnhancingPrompt(true)
    let cancelled = false
    const poll = async () => {
      if (cancelled) return
      const result = await ApiClient.getGenerationProgress()
      if (cancelled) return
      if (!result.ok) {
        setTimeout(poll, 2000)
        return
      }

      // The server's own "nothing running" signal: no reservation, no active generation. If the
      // marker was written but the backend never actually started a matching generation (e.g.
      // reload raced the request, or the backend restarted before picking it up), this is what
      // proves it — no need to guess from id comparisons.
      if (result.data.status === 'idle') {
        logger.warn(`Enhance recovery for ${ctx.projectId}: server reports idle, no generation ever started — dropping stale marker`)
        localStorage.removeItem(GENERATION_RECOVERY_KEY)
        setIsEnhancingPrompt(false)
        return
      }

      const observedId = result.data.id

      // Same identity confirmation as checkAndConsumeRecovery (lib/generation-recovery.ts): the
      // enhance endpoint's own pipeline/provider setup can take a moment before it ever reports
      // a new id, so an unconfirmed poll can still be looking at whatever the single global
      // progress slot held before this marker was even written — trusting that blindly can
      // misapply a stale, unrelated result (even a video path) as an "enhanced prompt".
      if (ctx.generationId == null) {
        if (observedId === ctx.baselineId) {
          setTimeout(poll, 2000)
          return
        }
        ctx = { ...ctx, generationId: observedId ?? undefined }
        localStorage.setItem(GENERATION_RECOVERY_KEY, JSON.stringify(ctx))
      } else if (observedId !== ctx.generationId) {
        // A different generation superseded ours before we ever saw it finish.
        logger.warn(`Enhance recovery: id changed from ${ctx.generationId} to ${observedId} before we saw it finish — dropping marker`)
        localStorage.removeItem(GENERATION_RECOVERY_KEY)
        setIsEnhancingPrompt(false)
        return
      }

      if (result.data.status === 'running') {
        setTimeout(poll, 2000)
        return
      }

      logger.info(`Enhance recovery: generation ${ctx.generationId} settled with status=${result.data.status}, clearing marker`)
      localStorage.removeItem(GENERATION_RECOVERY_KEY)
      setIsEnhancingPrompt(false)
      if (result.data.status === 'complete' && typeof result.data.result === 'string') {
        applyEnhanceResult(ctx.prompt, result.data.result)
      }
    }
    void poll()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // mount only

  const handleGenerate = async () => {
    if (mode === 'ic-lora') {
      if ((!prompt.trim() && !promptOptional) || !icLoraInput.videoPath || !icLoraInput.ready) return

      // Catalog IC-LoRA mode: the backend resolves catalog weights and builds the control
      // video from the user's input via the IC-LoRA's preprocessing pipeline.
      if (isCatalogIcLora && selectedIcLora) {
        icLoraSubmissionRef.current = {
          prompt,
          input: {
            videoPath: icLoraInput.videoPath,
            conditioningType: 'custom',
            conditioningStrength: icLoraStrength,
          },
        }
        await writeRecoveryContext({ prompt })
        await submitIcLora({
          videoPath: '',
          conditioningType: 'custom',
          conditioningStrength: icLoraStrength,
          prompt,
          icLoraId: selectedIcLora.id,
          variantId: selectedIcLoraVariantId ?? undefined,
          inputPath: icLoraInput.videoPath,
          controlValues,
          outpaintPads: hasPositionCanvas ? outpaintPads : undefined,
          allowEmptyPrompt: promptOptional,
          referenceImagePath: icLoraInput.referenceImagePath ?? undefined,
          // v1 backend reads the IC-LoRA's default_settings; overrides are sent only when the
          // advanced flag is on. _resolve_settings applies skipStage2/resolutionFactor/audioMode/
          // loraStrength server-side; fpsOverride is the exception — sent but ignored on the catalog path.
          ...(advancedIcLoraControls
            ? {
                skipStage2: icLoraSkipStage2,
                useLoraInStage2: icLoraUseLoraInStage2,
                resolution: resolveResolution(icLoraResolutionOpts, icLoraResolutionKey),
                resolutionFactor: icLoraResolutionFactor,
                audioMode: icLoraAudioMode,
                loraStrength: icLoraLoraStrength,
                fpsOverride: icLoraFps ?? undefined,
              }
            : {}),
        })
        return
      }

      // Flag may have flipped off after 'custom' was selected; don't submit a custom request then.
      if (icLoraCondType === 'custom' && !isCustomIcLoraEnabled) return
      const isCustomIcLora = icLoraCondType === 'custom'
      // Custom mode needs a selected IC-LoRA; the imported video is the control video.
      if (isCustomIcLora && !icLoraCustomRef) return
      icLoraSubmissionRef.current = {
        prompt,
        input: {
          videoPath: icLoraInput.videoPath,
          conditioningType: icLoraCondType,
          conditioningStrength: icLoraStrength,
          customLoraRef: isCustomIcLora ? icLoraCustomRef ?? undefined : undefined,
        },
      }
      await writeRecoveryContext({ prompt })
      await submitIcLora({
        videoPath: icLoraInput.videoPath,
        conditioningType: icLoraCondType,
        conditioningStrength: icLoraStrength,
        prompt,
        customLoraRef: isCustomIcLora ? icLoraCustomRef ?? undefined : undefined,
        controlVideoPath: isCustomIcLora ? icLoraInput.videoPath : undefined,
        skipStage2: icLoraSkipStage2,
        useLoraInStage2: icLoraUseLoraInStage2,
        resolution: resolveResolution(icLoraResolutionOpts, icLoraResolutionKey),
        resolutionFactor: icLoraResolutionFactor,
        audioMode: icLoraAudioMode,
        loraStrength: icLoraLoraStrength,
        fpsOverride: icLoraFps ?? undefined,
      })
      return
    }

    if (mode === 'retake') {
      if (!retakeInput.videoPath || retakeInput.duration < 2) return
      retakeSubmissionRef.current = {
        prompt,
        input: {
          videoPath: retakeInput.videoPath,
          startTime: retakeInput.startTime,
          duration: retakeInput.duration,
          videoDuration: retakeInput.videoDuration,
        },
      }
      await writeRecoveryContext({ prompt })
      await submitRetake({
        videoPath: retakeInput.videoPath,
        startTime: retakeInput.startTime,
        duration: retakeInput.duration,
        prompt,
        mode: 'replace_audio_and_video',
        resolution: resolveResolution(retakeResolutionOpts, retakeResolutionKey),
      })
      return
    }

    if (mode === 'extend') {
      if (!extendInput.videoPath || !extendInput.ready) return
      extendSubmissionRef.current = {
        prompt,
        input: {
          videoPath: extendInput.videoPath,
          direction: extendDirection,
          duration: extendSeconds,
          videoDuration: extendInput.videoDuration,
        },
      }
      await writeRecoveryContext({ prompt })
      await submitExtend({
        videoPath: extendInput.videoPath,
        duration: extendSeconds,
        prompt,
        mode: extendDirection,
        resolution: resolveResolution(extendResolutionOpts, extendResolutionKey),
      })
      return
    }

    if (!prompt.trim()) return

    // Save the prompt before generation starts
    setLastPrompt(prompt)

    if (mode === 'image') {
      const editSource = inputImage || null
      lastImageEditRef.current = editSource
        ? { source: editSource, strength: settings.imageEditStrength ?? 0.6 }
        : null
      const imageSettings = {
        model: 'fast' as 'fast' | 'pro',
        duration: 5,
        videoResolution: settings.videoResolution,
        fps: 24,
        audio: false,
        cameraMotion: 'none',
        imageResolution: settings.imageResolution,
        imageAspectRatio: settings.aspectRatio,
        imageSteps: editSource ? IMAGE_STEPS_EDIT : IMAGE_STEPS_GENERATE,
        variations: settings.variations,
        imageEditStrength: settings.imageEditStrength,
      }
      await writeRecoveryContext({ prompt, settings: imageSettings, genType: 'image', inputImageUrl: editSource ?? undefined })
      generateImage(prompt, imageSettings, editSource)
    } else {
      // Generate video (t2v if no image/audio, i2v if image, a2v if audio)
      const imagePath = inputImage || null
      const audioPath = inputAudio || null
      const videoSettings = sanitizeVideoSettings(settings)
      const genSettings = {
          model: videoSettings.model as 'fast' | 'pro',
          duration: videoSettings.duration,
          videoResolution: videoSettings.videoResolution,
          fps: videoSettings.fps,
          audio: videoSettings.audio || false,
          cameraMotion: 'none',
          aspectRatio: videoSettings.aspectRatio,
          imageResolution: videoSettings.imageResolution,
          imageAspectRatio: videoSettings.aspectRatio,
          imageSteps: 4,
          // Local LoRA refs are filesystem paths the cloud API can't resolve.
          loras: isLocalMode && selectedLoras.length > 0 ? selectedLoras : undefined,
      }
      await writeRecoveryContext({
        prompt,
        settings: genSettings,
        inputImageUrl: imagePath ?? undefined,
        inputAudioUrl: audioPath ?? undefined,
      })
      generate(prompt, imagePath, genSettings, audioPath)
    }
  }
  
  const handleDelete = (assetId: string) => {
    if (currentProjectId) {
      deleteAsset(currentProjectId, assetId)
    }
  }
  
  const handleDragStart = (e: React.DragEvent, asset: Asset) => {
    e.dataTransfer.setData('asset', JSON.stringify(asset))
    e.dataTransfer.setData('assetId', asset.id)
    e.dataTransfer.effectAllowed = 'copy'
  }
  
  const handleCreateVideo = (imageAsset: Asset) => {
    setMode('video')
    setInputImage(imageAsset.path)
    setPrompt(`${imageAsset.prompt || 'The scene comes to life...'}`)
  }

  const handleEditImage = (imageAsset: Asset) => {
    setMode('image')
    setInputImage(imageAsset.path)
    setPrompt((prev) => (prev.trim() ? prev : imageAsset.prompt || ''))
  }

  const handleRetake = (videoAsset: Asset) => {
    setMode('retake')
    setPrompt('')
    setActiveRetakeSource(null)
    setRetakeResolutionKey('original')
    setRetakeInitial({
      videoPath: videoAsset.path,
      duration: videoAsset.duration,
    })
    setRetakePanelKey((prev) => prev + 1)
  }

  const handleExtend = (videoAsset: Asset) => {
    setMode('extend')
    setPrompt('')
    setExtendDirection('end')
    setExtendSeconds(DEFAULT_EXTEND_SECONDS)
    setExtendResolutionKey('original')
    setExtendInitial({
      videoPath: videoAsset.path,
      duration: videoAsset.duration,
    })
    setExtendPanelKey((prev) => prev + 1)
  }

  const handleIcLora = (videoAsset: Asset) => {
    if (forceApiGenerations) return
    setMode('ic-lora')
    setPrompt('')
    setActiveIcLoraSource(null)
    setIcLoraInitial({ videoPath: videoAsset.path })
    setIcLoraPanelKey((prev) => prev + 1)
  }

  const isRetakeMode = mode === 'retake'
  const isExtendMode = mode === 'extend'
  const isIcLoraMode = mode === 'ic-lora'
  const hasCompatibleVideoSettings = mode !== 'video' || (
    !isLoadingVideoGenerationModelSpecs
    && videoModelSpecs.length > 0
    && resolveVideoGenerationOptions({
      settings,
      modelSpecs: videoModelSpecs,
      hasAudio: Boolean(inputAudio),
    }).hasCompatibleOptions
  )
  const canSubmit = !isOtherGenerationRunning && (isRetakeMode
    ? retakeInput.ready && !!retakeInput.videoPath && !isRetaking
    : isExtendMode
      ? extendInput.ready && !!extendInput.videoPath && !isExtending
      : isIcLoraMode
        ? (!!prompt.trim() || promptOptional) && icLoraInput.ready && !!icLoraInput.videoPath && !isIcLoraGenerating
          && (isCatalogIcLora || icLoraCondType !== 'custom' || !!icLoraCustomRef)
        : !!prompt.trim() && hasCompatibleVideoSettings)
  const promptButtonLabel = isRetakeMode ? 'Retake' : isExtendMode ? 'Extend' : isIcLoraMode ? 'Generate' : 'Generate'
  const promptButtonIcon = isRetakeMode
    ? <Scissors className="h-3.5 w-3.5" />
    : isExtendMode
      ? <MoveHorizontal className="h-3.5 w-3.5" />
      : isIcLoraMode
        ? <Sparkles className="h-3.5 w-3.5" />
    : <Sparkles className={`h-3.5 w-3.5 ${isGenerating ? 'animate-pulse' : ''}`} />
  const promptGenerating = isRetakeMode ? isRetaking : isExtendMode ? isExtending : isIcLoraMode ? isIcLoraGenerating : isGenerating
  
  // Close size menu on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (sizeMenuRef.current && !sizeMenuRef.current.contains(e.target as Node)) {
        setShowSizeMenu(false)
      }
    }
    if (showSizeMenu) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [showSizeMenu])

  const filteredAssets = showFavorites ? assets.filter(a => a.favorite) : assets
  const favoriteCount = assets.filter(a => a.favorite).length
  const isLibraryMode = mode === 'video' || mode === 'image'

  // Navigation for the asset preview modal
  const selectedIndex = selectedAsset ? filteredAssets.findIndex(a => a.id === selectedAsset.id) : -1
  const canGoPrev = selectedIndex > 0
  const canGoNext = selectedIndex >= 0 && selectedIndex < filteredAssets.length - 1

  const goToPrev = useCallback(() => {
    if (canGoPrev) setSelectedAsset(filteredAssets[selectedIndex - 1])
  }, [canGoPrev, filteredAssets, selectedIndex])

  const goToNext = useCallback(() => {
    if (canGoNext) setSelectedAsset(filteredAssets[selectedIndex + 1])
  }, [canGoNext, filteredAssets, selectedIndex])

  // Keyboard navigation for the preview modal
  useEffect(() => {
    if (!selectedAsset) return
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') { e.preventDefault(); goToPrev() }
      else if (e.key === 'ArrowRight') { e.preventDefault(); goToNext() }
      else if (e.key === 'Escape') setSelectedAsset(null)
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [selectedAsset, goToPrev, goToNext])

  // Shared IC-LoRA control props — consumed by the bottom-row settings (PromptBar) and the
  // advanced side panel beside the prompt.
  const icLoraControlsProps: IcLoraControlsProps = {
    icLoraCondType,
    icLoraSelectorValue,
    icLoraSelectorOptions,
    onIcLoraSelectorChange: handleIcLoraSelectorChange,
    icLoraStrength,
    onIcLoraStrengthChange: setIcLoraStrength,
    availableIcLoras: installedIcLoras,
    icLoraCustomRef,
    onIcLoraCustomRefChange: setIcLoraCustomRef,
    icLoraSkipStage2,
    onIcLoraSkipStage2Change: setIcLoraSkipStage2,
    icLoraUseLoraInStage2,
    onIcLoraUseLoraInStage2Change: setIcLoraUseLoraInStage2,
    icLoraResolutionOptions: icLoraResolutionOpts,
    icLoraResolutionKey,
    onIcLoraResolutionKeyChange: setIcLoraResolutionKey,
    icLoraResolutionFactor,
    onIcLoraResolutionFactorChange: setIcLoraResolutionFactor,
    icLoraAudioMode,
    onIcLoraAudioModeChange: setIcLoraAudioMode,
    icLoraLoraStrength,
    onIcLoraLoraStrengthChange: setIcLoraLoraStrength,
    icLoraFps,
    onIcLoraFpsChange: setIcLoraFps,
    isCatalogIcLora,
    advancedIcLoraControls,
    controls: selectedIcLora?.controls ?? [],
    controlValues,
    onControlChange,
  }

  return (
    <div className="h-full relative bg-zinc-950">

      {/* Empty state */}
      {isLibraryMode && assets.length === 0 && !isGenerating && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none">
          <div className="w-24 h-24 rounded-2xl border-2 border-dashed border-zinc-700 flex items-center justify-center mb-4">
            <Sparkles className="h-10 w-10 text-zinc-600" />
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">Start Creating</h3>
          <p className="text-zinc-500 max-w-md">
            Use the prompt bar below to generate images and videos.
            Drag assets into the input box to use them as references.
          </p>
        </div>
      )}

      {/* No favorites empty state */}
      {isLibraryMode && showFavorites && filteredAssets.length === 0 && assets.length > 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none">
          <Heart className="h-12 w-12 text-zinc-700 mb-4" />
          <h3 className="text-lg font-semibold text-white mb-2">No favorites yet</h3>
          <p className="text-zinc-500 text-sm">
            Click the heart icon on any asset to add it to your favorites.
          </p>
        </div>
      )}

      {/* Assets area — full width, no background, above the prompt bar */}
      {/* Kept mounted even with no assets so the Browse LoRAs / Favorites / size toolbar survives the empty state. */}
      {isLibraryMode && (
        <div className="absolute inset-x-0 top-0 bottom-[160px] flex flex-col px-4 pt-4">
          {/* Top bar */}
          <div className="flex items-center justify-between pb-2 gap-2">
            <div className="flex items-center gap-2">
              {mode === 'video' && isLocalMode && (
                <button
                  onClick={() => loraLibrary.setModalOpen(true)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
                >
                  <Sparkles className="h-4 w-4" /> Browse LoRAs
                </button>
              )}
            </div>
            <div className="flex items-center gap-2">
            <button
              onClick={() => setShowFavorites(!showFavorites)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                showFavorites
                  ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                  : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
              }`}
            >
              <Heart className={`h-4 w-4 ${showFavorites ? 'fill-current' : ''}`} />
              Favorites
              {favoriteCount > 0 && (
                <span className={`text-xs px-1.5 py-0.5 rounded-full ${
                  showFavorites ? 'bg-red-500/30 text-red-300' : 'bg-zinc-800 text-zinc-500'
                }`}>
                  {favoriteCount}
                </span>
              )}
            </button>

            <div ref={sizeMenuRef} className="relative">
              <button
                onClick={() => setShowSizeMenu(!showSizeMenu)}
                className={`p-2 rounded-md transition-colors ${
                  showSizeMenu ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
                }`}
              >
                {gallerySize === 'small' ? <GridSmallIcon className="h-4 w-4" /> :
                 gallerySize === 'medium' ? <GridMediumIcon className="h-4 w-4" /> :
                 <GridLargeIcon className="h-4 w-4" />}
              </button>

              {showSizeMenu && (
                <div className="absolute top-full mt-2 right-0 bg-zinc-800 border border-zinc-700 rounded-md p-2 min-w-[160px] shadow-xl z-50">
                  {([
                    { value: 'small' as GallerySize, label: 'Small', icon: GridSmallIcon },
                    { value: 'medium' as GallerySize, label: 'Medium', icon: GridMediumIcon },
                    { value: 'large' as GallerySize, label: 'Large', icon: GridLargeIcon },
                  ]).map(option => (
                    <button
                      key={option.value}
                      onClick={() => { setGallerySize(option.value); setShowSizeMenu(false) }}
                      className={`w-full flex items-center justify-between px-2 py-2.5 rounded-md transition-colors text-left ${gallerySize === option.value ? 'bg-white/20 hover:bg-white/25' : 'hover:bg-zinc-700'}`}
                    >
                      <div className="flex items-center gap-3">
                        <option.icon className={`h-4 w-4 ${gallerySize === option.value ? 'text-white' : 'text-zinc-500'}`} />
                        <span className={`text-sm ${gallerySize === option.value ? 'text-white font-medium' : 'text-zinc-400'}`}>
                          {option.label}
                        </span>
                      </div>
                      {gallerySize === option.value && (
                        <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </div>
            </div>
          </div>

          {/* Assets grid — fills remaining space, scrollable */}
          <div className="overflow-y-auto overflow-x-hidden [scrollbar-gutter:stable] flex-1">
            <div className={`grid ${gallerySizeClasses[gallerySize]} gap-4`}>
              {isGenerating && (
                <div className="relative rounded-xl overflow-hidden bg-zinc-800 aspect-video">
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <div className="relative w-16 h-16 mb-3">
                      <div className="absolute inset-0 rounded-full border-2 border-violet-500/30" />
                      <div className="absolute inset-0 rounded-full border-2 border-violet-500 border-t-transparent animate-spin" />
                      <div className="absolute inset-2 rounded-full bg-zinc-800 flex items-center justify-center">
                        <Sparkles className="h-6 w-6 text-violet-400" />
                      </div>
                    </div>
                    <p className="text-sm text-zinc-400">{statusMessage || 'Generating...'}</p>
                    {progress > 0 && (
                      <div className="w-32 h-1 bg-zinc-800 rounded-full mt-2 overflow-hidden">
                        <div className="h-full bg-violet-500 transition-all" style={{ width: `${progress}%` }} />
                      </div>
                    )}
                  </div>
                </div>
              )}
              {filteredAssets.map(asset => (
                <AssetCard
                  key={asset.id}
                  asset={asset}
                  onDelete={() => handleDelete(asset.id)}
                  onPlay={() => setSelectedAsset(asset)}
                  onDragStart={handleDragStart}
                  onCreateVideo={handleCreateVideo}
                  onEditImage={handleEditImage}
                  onRetake={handleRetake}
                  onExtend={handleExtend}
                  onIcLora={!forceApiGenerations ? handleIcLora : undefined}
                  onToggleFavorite={() => currentProjectId && toggleFavorite(currentProjectId, asset.id)}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      {mode === 'retake' && (
        <div className="absolute inset-x-0 top-0 bottom-[160px] px-4 pt-4 pb-4 flex flex-col overflow-hidden">
          <RetakePanel
            initialVideoPath={retakeInitial.videoPath}
            initialDuration={retakeInitial.duration}
            resetKey={retakePanelKey}
            fillHeight
            isProcessing={isRetaking}
            processingStatus={retakeStatus}
            enforceApiConstraints={!isLocalMode}
            onChange={setRetakeInput}
          />
        </div>
      )}

      {mode === 'extend' && (
        <div className="absolute inset-x-0 top-0 bottom-[160px] px-4 pt-4 pb-4 flex flex-col overflow-hidden">
          <ExtendPanel
            initialVideoPath={extendInitial.videoPath}
            initialDuration={extendInitial.duration}
            resetKey={extendPanelKey}
            fillHeight
            isProcessing={isExtending}
            processingStatus={extendStatus}
            enforceApiConstraints={!isLocalMode}
            onChange={setExtendInput}
          />
        </div>
      )}

      {mode === 'ic-lora' && !forceApiGenerations && (
        // Extra bottom clearance (vs 160px elsewhere): the floating panel here also carries the
        // selected-IC-LoRA info banner, so reserve room so it can't cover the panel's bottom bar.
        <div className="absolute inset-x-0 top-0 bottom-[210px] px-4 pt-4 pb-4 flex flex-col overflow-hidden">
          <ICLoraPanel
            initialVideoPath={icLoraInitial.videoPath}
            resetKey={icLoraPanelKey}
            fillHeight
            isProcessing={isIcLoraGenerating}
            processingStatus={icLoraStatus}
            inputKind={selectedIcLora?.input?.kind ?? 'video'}
            selectedIcLoraId={selectedIcLoraId}
            allowsReferenceImage={selectedIcLora?.allows_reference_image ?? false}
            showOutpaintCanvas={hasPositionCanvas}
            outpaintPads={outpaintPads}
            onOutpaintPadsChange={setOutpaintPads}
            isLocalMode={isLocalMode}
            onBrowseLibrary={() => setLibraryModalOpen(true)}
            conditioningType={icLoraCondType}
            onConditioningTypeChange={handleIcLoraCondTypeChange}
            conditioningStrength={icLoraStrength}
            onConditioningStrengthChange={setIcLoraStrength}
            outputVideoPath={icLoraResult?.videoPath || null}
            onChange={setIcLoraInput}
          />
          <LoraLibraryModal
            open={libraryModalOpen}
            onClose={() => setLibraryModalOpen(false)}
            kind="ic-lora"
            items={icLoraItems}
            selectedId={selectedIcLoraId}
            selectedVariantId={selectedIcLoraVariantId}
            downloadingKey={downloadingIcLoraKey}
            progress={icLoraDownloadProgress}
            downloadError={icLoraDownloadError}
            onDownload={downloadIcLora}
            onSelect={(e, variantId) => {
              selectIcLora(icLoras.find(r => r.ic_lora.id === e.id) ?? null, variantId)
              return true
            }}
          />
        </div>
      )}

      {/* Floating prompt panel — wider, responsive, centered */}
      <div className="absolute bottom-5 left-1/2 w-[min(860px,calc(100%-2rem))] -translate-x-1/2">

        <FreeApiKeyBubble
          forceApiGenerations={forceApiGenerations}
          hasLtxApiKey={appSettings.hasLtxApiKey}
          isGenerating={isGenerating}
        />

        {/* Active IC-LoRA (IC-LoRA view only). */}
        {mode === 'ic-lora' && isCatalogIcLora && selectedIcLora && (
          <SelectedLoraInfo items={[{
            name: variantDisplayName(
              selectedIcLora.name,
              selectedIcLoraVariantId
                ? selectedIcLora.download.variants?.find(v => v.id === selectedIcLoraVariantId)?.label
                : undefined,
              selectedIcLora.download.variants?.length,
            ),
            sections: selectedIcLora.instructions ?? undefined,
            repoId: selectedIcLora.download.repo_id,
            isCommunity: selectedIcLora.author?.affiliation !== 'ltx',
          }]} />
        )}

        {/* Selected plain LoRAs (video gen only) — one chip each, info from the matched catalog entry. */}
        {mode === 'video' && isLocalMode && selectedLoras.length > 0 && (
          <SelectedLoraInfo items={selectedLoras.map(s => {
            const entry = loraLibrary.items.find(e => e.installedPath === s.ref)
            return {
              name: s.name,
              sections: entry?.instructions,
              repoId: entry?.repoId,
              isCommunity: entry?.author?.affiliation !== 'ltx',
            }
          })} />
        )}

        {/* Prompt bar */}
        <PromptBar
          mode={mode}
          onModeChange={setMode}
          canUseIcLora={!forceApiGenerations}
          prompt={prompt}
          onPromptChange={setPrompt}
          onGenerate={handleGenerate}
          isGenerating={promptGenerating}
          canGenerate={canSubmit}
          buttonLabel={promptButtonLabel}
          buttonIcon={promptButtonIcon}
          extendDirection={extendDirection}
          onExtendDirectionChange={setExtendDirection}
          extendSeconds={extendSeconds}
          onExtendSecondsChange={setExtendSeconds}
          resolutionOptions={mode === 'extend' ? extendResolutionOpts : mode === 'retake' ? retakeResolutionOpts : []}
          selectedResolution={mode === 'extend' ? extendResolutionKey : retakeResolutionKey}
          onResolutionChange={mode === 'extend' ? setExtendResolutionKey : setRetakeResolutionKey}
          inputImage={inputImage}
          onInputImageChange={setInputImage}
          inputAudio={inputAudio}
          onInputAudioChange={setInputAudio}
          settings={settings}
          onSettingsChange={(nextSettings) => setSettings(sanitizeVideoSettings(nextSettings))}
          videoModelSpecs={videoModelSpecs}
          videoSettingsMessage={videoSettingsMessage}
          icLoraControls={icLoraControlsProps}
          promptOptional={promptOptional}
          isLocalMode={isLocalMode}
          imageUsesFalApi={shouldImageGenerateWithFalApi}
          availableLoras={localLoras}
          selectedLoras={selectedLoras}
          onSelectedLorasChange={setSelectedLoras}
          loraDisplayNames={loraDisplayNames}
          loraCatalogIdsByPath={loraCatalogIdsByPath}
          enhanceProviderAvailable={isEnhancerProviderAvailable}
          enhanceAvailableForMode={enhanceAvailableForMode}
          canEnhancePrompt={canEnhancePrompt}
          enhanceProvider={enhanceProvider}
          onEnhanceProviderChange={canToggleEnhanceProvider ? setEnhanceProviderPref : undefined}
          isEnhancingPrompt={isEnhancingPrompt}
          enhancePromptError={enhancePromptError}
          onEnhancePrompt={handleEnhancePrompt}
          canUndoPrompt={canUndoPrompt}
          onUndoPrompt={handleUndoPrompt}
          canRedoPrompt={canRedoPrompt}
          onRedoPrompt={handleRedoPrompt}
        />

        {/* Advanced IC-LoRA controls — bottom-aligned to the right of the prompt panel. */}
        {mode === 'ic-lora' && !forceApiGenerations && advancedIcLoraControls && (
          <div className="absolute left-full bottom-0 ml-3">
            <IcLoraAdvancedPanel {...icLoraControlsProps} />
          </div>
        )}
      </div>

      <LoraLibraryModal
        open={loraLibrary.modalOpen}
        onClose={() => loraLibrary.setModalOpen(false)}
        kind="lora"
        items={loraLibrary.items}
        selectedId={null}
        downloadingKey={loraLibrary.downloadingKey}
        progress={loraLibrary.progress}
        downloadError={loraLibrary.downloadError}
        syncError={loraLibrary.useError}
        onDownload={loraLibrary.downloadLora}
        onSelect={loraLibrary.useEntry}
      />

      {/* Asset preview modal */}
      {selectedAsset && (
        <div 
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center"
          onClick={() => setSelectedAsset(null)}
        >
          {/* Previous button */}
          <button
            onClick={(e) => { e.stopPropagation(); goToPrev() }}
            disabled={!canGoPrev}
            className={`absolute left-4 top-1/2 -translate-y-1/2 z-10 p-3 rounded-full backdrop-blur-md transition-all ${
              canGoPrev
                ? 'bg-white/10 text-white hover:bg-white/20 cursor-pointer'
                : 'bg-white/5 text-zinc-600 cursor-default'
            }`}
          >
            <ChevronLeft className="h-6 w-6" />
          </button>

          {/* Next button */}
          <button
            onClick={(e) => { e.stopPropagation(); goToNext() }}
            disabled={!canGoNext}
            className={`absolute right-4 top-1/2 -translate-y-1/2 z-10 p-3 rounded-full backdrop-blur-md transition-all ${
              canGoNext
                ? 'bg-white/10 text-white hover:bg-white/20 cursor-pointer'
                : 'bg-white/5 text-zinc-600 cursor-default'
            }`}
          >
            <ChevronRight className="h-6 w-6" />
          </button>

          {/* Content area */}
          <div className="relative max-w-5xl w-full max-h-full px-20 py-8" onClick={e => e.stopPropagation()}>
            {/* Top bar: counter + close */}
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm text-zinc-500 font-medium">
                {selectedIndex + 1} / {filteredAssets.length}
              </span>
              <button
                onClick={() => setSelectedAsset(null)}
                className="p-2 rounded-md text-zinc-400 hover:text-white transition-colors"
              >
                <X className="h-6 w-6" />
              </button>
            </div>

            {selectedAsset.type === 'video' ? (
              <video
                key={selectedAsset.id}
                src={pathToFileUrl(selectedAsset.path)}
                controls
                autoPlay
                className="w-full rounded-xl object-contain max-h-[75vh]"
              />
            ) : (
              <img
                key={selectedAsset.id}
                src={pathToFileUrl(selectedAsset.path)}
                alt=""
                className="w-full rounded-xl object-contain max-h-[75vh]"
              />
            )}
            <div className="mt-4 text-center">
              <div className="inline-flex items-start gap-2 max-w-full">
                <p className="text-zinc-300 max-h-40 overflow-y-auto whitespace-pre-wrap break-words text-left">{selectedAsset.prompt}</p>
                {selectedAsset.prompt && (
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(selectedAsset.prompt)
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
              <p className="text-zinc-500 text-sm mt-1">
                {[
                  selectedAsset.resolution,
                  selectedAsset.duration ? `${formatSeconds(selectedAsset.duration)}s` : 'Image',
                ].filter(Boolean).join(' • ')}
              </p>
            </div>
          </div>
        </div>
      )}

      {(error || localError) && (
        <GenerationErrorDialog
          error={(error || localError)!}
          onDismiss={() => {
            if (error) reset()
            if (localError) {
              setLocalError(null)
              resetRetake()
              resetIcLora()
            }
          }}
        />
      )}
    </div>
  )
}
