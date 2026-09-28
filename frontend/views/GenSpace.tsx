import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { createPortal } from 'react-dom'
import {
  Trash2, Download, Image, Video, X,
  Heart, Film, Volume2, VolumeX, Sparkles, Sparkle,
  Clock, Monitor, ChevronUp, Scissors, Music, Undo2, Redo2, Loader2,
  ChevronLeft, ChevronRight, Copy, Check, Tag, Eraser, Square, MoveHorizontal, Wand2, Rows3, RefreshCw, Clapperboard,
  CheckSquare, Lock, Dices, History, Shuffle
} from 'lucide-react'
import { useProjects } from '../contexts/ProjectContext'
import type { GenSpaceRetakeSource } from '../contexts/ProjectContext'
import { useAppSettings } from '../contexts/AppSettingsContext'
import { useGeneration, GENERATION_RECOVERY_KEY, type GenerationRecoveryContext } from '../hooks/use-generation'
import { useFixedMenu } from '../hooks/use-fixed-menu'
import { setActiveGenerationOwner, hasValidBaselineId } from '../lib/generation-recovery'
import { withGenerationActive, canCancelLocalJob } from '../lib/generation-active'
import { useVideoGenerationModelSpecs } from '../hooks/use-video-generation-model-specs'
import { createLocalGenerationError, type GenerationError } from '../lib/generation-errors'
import {
  useRetake,
  RETAKE_EXTEND_MODELS,
  retakeExtendModelFromPipeline,
  type RetakeExtendModel,
} from '../hooks/use-retake'
import { useExtend, type ExtendDirection, EXTEND_SECONDS, DEFAULT_EXTEND_SECONDS } from '../hooks/use-extend'
import { resolutionOptions, videoGenerationResolutionLabel, type ResolutionOption } from '../lib/video-resolution'
import { useIcLora, type IcLoraAudioMode } from '../hooks/use-ic-lora'
import { useCustomIcLoraEnabled } from '../hooks/use-custom-ic-lora-enabled'
import { useDevFlags } from '../contexts/DevFlagsContext'
import { type IcLoraListItem } from '../hooks/use-catalog'
import { useIcLoraLibrary } from '../hooks/use-ic-lora-library'
import { useLoraLibrary } from '../hooks/use-lora-library'
import { usePromptEnhancerProvider, type EnhanceProvider } from '../hooks/use-prompt-enhancer-provider'
import { useGlobalGenerationLock } from '../hooks/use-global-generation-lock'
import type { ICLoraConditioningType } from '../components/ICLoraPanel'
import type { Asset } from '../types/project-model'
import { GenerationErrorDialog } from '../components/GenerationErrorDialog'
import { addVisualAssetToProject } from '../lib/asset-copy'
import { pathToFileUrl } from '../lib/file-url'
import { GPM_IMAGE_DND_TYPE, saveDataUrlToTempFile, type GpmDndImage } from '../components/gpm/gpm-image-file'
import { FILE_DND, type LibFile } from '../components/gpm/DownloadsBrowser'
import {
  areVideoGenerationSettingsEquivalent,
  formatPipelineDisplayName,
  GENSPACE_MIN_SELECTABLE_DURATION_S,
  getApiOfferingCapabilities,
  getVideoGenerationModelSpecs,
  getLocalOfferingCapabilities,
  resolvePipelineDisplayName,
  resolveVideoGenerationOptions,
  sanitizeVideoGenerationSettings,
  type VideoGenerationModelSpecItem,
  type VideoGenerationPipeline,
} from '../lib/video-generation-model-specs'
import { logger } from '../lib/logger'
import { ApiClient, type ApiSuccessOf } from '../lib/api-client'
import type { GenerationSettings, LoraSelection } from '../components/SettingsPanel'
import { RetakePanel } from '../components/RetakePanel'
import { ExtendPanel } from '../components/ExtendPanel'
import { MultiKeyframePanel } from '../components/MultiKeyframePanel'
import { ICLoraPanel, CONDITIONING_TYPES } from '../components/ICLoraPanel'
import type { OutpaintPads } from '../components/OutpaintCanvasEditor'
import { LoraLibraryModal } from '../components/LoraLibraryModal'
import { SelectedLoraInfo } from '../components/SelectedLoraInfo'
import { buildIcLoraSelectorOptions, encodeIcLoraSelectorValue, parseIcLoraSelectorValue, toModelsDirRelativeRef, variantDisplayName } from '../lib/lora-library'
import { SettingsDropdown } from '../components/SettingsDropdown'
import { IcLoraSettingsControls, type IcLoraControlsProps } from '../components/IcLoraSettingsControls'
import { IcLoraAdvancedPanel } from '../components/IcLoraAdvancedPanel'
import { FreeApiKeyBubble } from '../components/FreeApiKeyBubble'
import { shouldShowLastFrameChip } from '../lib/genspace-last-frame'
import {
  GEMINI_KEY_REQUIRED_SETTINGS_DETAIL,
  isEnhanceBlockedByMissingGeminiKey,
} from '../lib/enhance-gemini-key'
import {
  autoDurationOptionVisible,
  canUseMultiKeyframeMode,
  fallbackGenSpaceMode,
  genSpaceUsesAudioInput,
  isEnhanceAvailableForMode,
  modeAfterCompletedGeneration,
  modeOptionValues,
} from '../lib/genspace-multi-keyframe'
import {
  enhanceKeyframesPayload,
  fromPersistedKeyframes,
  LOCAL_MULTI_KEYFRAME_MAX_COUNT,
  toPersistedKeyframes,
  videoGenerationModeFromInputs,
  type KeyframeItem,
} from '../lib/multi-keyframe'
import { lastFrameFromDuration, retimeKeyframesForSettings, type DraggedFrame } from '../lib/keyframe-timeline'
import { useVideoSaveMenu } from '../components/useVideoSaveMenu'
import { useImageSaveMenu } from '../components/useImageSaveMenu'
import { saveToStudioAssets, saveVideoFile, saveImageFile } from '../lib/video-save-actions'

// Sentinel binFilter value meaning "show every asset" (vs. a real binId, or
// null for the default untagged-only view).
const ALL_BINS_FILTER = '__all__'

// Format a millisecond duration as a stopwatch clock, e.g. 0:07, 1:23.
function formatClock(ms: number): string {
  const totalSec = Math.max(0, Math.round(ms / 1000))
  const m = Math.floor(totalSec / 60)
  const s = totalSec % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

// Plain generations carry a complete recipe (prompt + source image + settings +
// LoRAs) in `generationParams`, so they can be restored into Gen Space wholesale.
// Retake/Extend/IC-LoRA depend on their source-video panel state instead, and
// keep using their own flows.
const REGENERABLE_MODES = new Set<string>([
  'text-to-video', 'image-to-video', 'audio-to-video', 'text-to-image', 'image-edit',
])
function canRegenerateAsset(asset: Asset): boolean {
  return !!asset.generationParams && REGENERABLE_MODES.has(asset.generationParams.mode)
}

// Asset card with hover overlays
function AssetCard({
  asset,
  onDelete,
  onPlay,
  onDragStart,
  onCreateVideo,
  onRegenerate,
  onEditImage,
  onRetake,
  onExtend,
  onContinue,
  onIcLora,
  onToggleFavorite,
  bins,
  onTag,
  onRequestNewTag,
  selectMode = false,
  selected = false,
  onToggleSelect,
}: {
  asset: Asset
  onDelete: () => void
  onPlay: () => void
  onDragStart: (e: React.DragEvent, asset: Asset) => void
  onCreateVideo?: (asset: Asset) => void
  onRegenerate?: (asset: Asset) => void
  onEditImage?: (asset: Asset) => void
  onRetake?: (asset: Asset) => void
  onExtend?: (asset: Asset) => void
  onContinue?: (asset: Asset) => void
  onIcLora?: (asset: Asset) => void
  onToggleFavorite?: () => void
  bins: Record<string, string>
  onTag: (binId: string | null) => void
  onRequestNewTag: () => void
  selectMode?: boolean
  selected?: boolean
  onToggleSelect?: () => void
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
    // Quick-save into the last-used Studio Assets folder (no native Save dialog).
    // Right-click the enlarged asset for the "Save…" menu to choose a location instead.
    // The prompt drives content-aware auto-naming (subject-NN.ext) in the library.
    void saveToStudioAssets(asset.path, asset.prompt)
  }

  // "Lightweight" post-to-X: no API keys/OAuth — opens X's compose page with
  // the prompt pre-filled as a caption, and reveals the file in Explorer so
  // the user can drag it straight into the compose box to attach it.
  const handlePostToX = (e: React.MouseEvent) => {
    e.stopPropagation()
    void window.electronAPI?.showItemInFolder({ filePath: asset.path })
    void window.electronAPI?.openTwitterCompose({ text: asset.prompt })
  }

  return (
    <div
      data-asset-id={asset.id}
      className={`relative group cursor-pointer rounded-xl overflow-hidden bg-zinc-900 ${
        selectMode && selected ? 'ring-2 ring-accent ring-offset-2 ring-offset-zinc-950' : ''
      }`}
      onMouseEnter={() => { if (!selectMode) setIsHovered(true) }}
      onMouseLeave={() => {
        setIsHovered(false)
        setCurrentTime(0)
      }}
      onClick={selectMode ? () => onToggleSelect?.() : onPlay}
      draggable={asset.type === 'image' && !selectMode}
      onDragStart={(e) => asset.type === 'image' && !selectMode && onDragStart(e, asset)}
    >
      {/* Selection checkbox overlay (multi-select mode) */}
      {selectMode && (
        <div className="absolute inset-0 z-30 flex items-start justify-start p-2 pointer-events-none">
          <div className={`flex h-6 w-6 items-center justify-center rounded-md border-2 transition-colors ${
            selected ? 'bg-accent border-accent text-white' : 'bg-black/50 border-white/70 text-transparent'
          }`}>
            <Check className="h-4 w-4" strokeWidth={3} />
          </div>
        </div>
      )}
      {/* In select mode, a full-card shield swallows hover controls so clicks only toggle selection. */}
      {selectMode && <div className="absolute inset-0 z-20" />}
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

      {/* Render time (local generations only) — parked bottom-left when idle so it
          reads at a glance without adding to the already-busy hover control row. */}
      {asset.renderMs != null && !isHovered && (
        <div
          className="absolute bottom-2 left-2 flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-black/50 backdrop-blur-md text-white/90 text-[10px] font-mono tabular-nums z-10 pointer-events-none"
          title="Render time"
        >
          <Clock className="h-2.5 w-2.5" />
          {formatClock(asset.renderMs)}
        </div>
      )}
      
      {/* Hover overlay. Held below full opacity so the controls read as a gentle
          layer over the media rather than a harsh bright bar. */}
      <div className={`absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-black/30 transition-opacity duration-200 ${
        isHovered ? 'opacity-75' : 'opacity-0'
      }`}>
        {/* Top buttons */}
        <div className="absolute top-2 left-2 right-2 flex items-center justify-between flex-wrap gap-y-1.5">
          {/* Left actions: originals stay on row 1; Regenerate sits on its own
              row below so it can't push IC-LoRA off a narrow card. */}
          <div className="flex flex-col items-start gap-1.5">
          <div className="flex items-center gap-1.5 flex-wrap">
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
                {onRetake && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onRetake(asset) }}
                    className="px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors flex items-center gap-1.5 text-xs font-medium whitespace-nowrap"
                  >
                    <Scissors className="h-3 w-3" />
                    Retake
                  </button>
                )}
                {onExtend && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onExtend(asset) }}
                    className="px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors flex items-center gap-1.5 text-xs font-medium whitespace-nowrap"
                  >
                    <MoveHorizontal className="h-3 w-3" />
                    Extend
                  </button>
                )}
                {onContinue && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onContinue(asset) }}
                    title="Continue as a new clip — seeds a fresh generation from this shot's last frame"
                    className="px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors flex items-center gap-1.5 text-xs font-medium whitespace-nowrap"
                  >
                    <Clapperboard className="h-3 w-3" />
                    Continue
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

          {onRegenerate && canRegenerateAsset(asset) && (
            <button
              onClick={(e) => { e.stopPropagation(); onRegenerate(asset) }}
              title="Load this generation's image, prompt and settings back into Create"
              className="px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors flex items-center gap-1.5 text-xs font-medium whitespace-nowrap"
            >
              <RefreshCw className="h-3 w-3" />
              Regenerate
            </button>
          )}
          </div>

          <div className="flex items-center gap-1.5">
            {asset.type === 'video' && (
              <button
                onClick={handlePostToX}
                title="Post to X — opens compose with your caption, reveals the file to drag in"
                className="p-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors"
              >
                <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="currentColor">
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                </svg>
              </button>
            )}
            <button
              onClick={handleDownload}
              className="p-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors"
            >
              <Download className="h-3.5 w-3.5" />
            </button>
            <div onClick={(e) => e.stopPropagation()}>
              <SettingsDropdown
                title="Tag"
                value={asset.binId ?? ''}
                onChange={(value) => {
                  if (value === '__new__') {
                    onRequestNewTag()
                  } else if (value === '__none__') {
                    onTag(null)
                  } else {
                    onTag(value)
                  }
                }}
                trigger={<Tag className={`h-3.5 w-3.5 ${asset.binId ? 'text-blue-300' : 'text-white'}`} />}
                options={[
                  ...(asset.binId ? [{ value: '__none__', label: 'Remove from folder' }] : []),
                  ...Object.entries(bins).map(([binId, name]) => ({ value: binId, label: name })),
                  { value: '__new__', label: '+ New Tag…' },
                ]}
              />
            </div>
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

// The IC-LoRA the mode picker defaults to (vs the built-in Canny Edges), when it's downloaded.
// Its optimal settings ride along via the catalog entry's default_settings (applied by
// onSelectIcLora), so there's nothing to hardcode here beyond the id.
const DEFAULT_IC_LORA_ID = 'ingredients'

const IMAGE_STEPS_GENERATE = 4
const IMAGE_STEPS_EDIT = 8
// FORK: Krea 2 Turbo (FLUX-family, ~4s/step in NF4) runs heavier than Z-Image.
// 6 steps trades a little detail for ~25% faster renders; tune to taste.
const IMAGE_STEPS_KREA2 = 6
// FORK: Krea 2 is dev-only (license requires deployer content filters + forbids removing
// its safety tuning); release builds offer Z-Image only. Backend enforces the same rule.
const KREA2_ENABLED = import.meta.env.DEV
type ImageModelId = 'z-image-turbo' | 'krea-2-turbo'
const allowedImageModel = (model: string | undefined): ImageModelId =>
  model === 'krea-2-turbo' && KREA2_ENABLED ? 'krea-2-turbo' : 'z-image-turbo'

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
  const { isOpen, setIsOpen, triggerRef, menuRef, style } = useFixedMenu('above')

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
    <div ref={triggerRef} className="relative">
      <button
        onClick={() => setIsOpen(o => !o)}
        className="flex items-center gap-1.5 px-2 py-1.5 rounded-md bg-zinc-800/60 text-zinc-300 text-xs hover:bg-zinc-700/60 transition-colors max-w-[160px]"
      >
        <Sparkles className="h-3.5 w-3.5 flex-shrink-0" />
        <span className="truncate">{label}</span>
      </button>
      {isOpen && createPortal(
        <div
          ref={menuRef}
          style={style}
          className="fixed w-72 max-h-80 overflow-y-auto rounded-md border border-zinc-700 bg-zinc-900 shadow-xl"
        >
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
        </div>,
        document.body,
      )}
    </div>
  )
}

type GenSpaceMode = 'image' | 'video' | 'retake' | 'extend' | 'ic-lora' | 'multi-keyframe'

// Resolve a selected resolution option key to {width,height}, or undefined for "original"
// (backend then uses the source resolution).
const MAX_SEED = 2147483647
const randomSeed = () => Math.floor(Math.random() * MAX_SEED)

// Quick seed controls in the prompt bar — same seedLocked/lockedSeed app settings as the
// Settings modal (the context debounce-syncs them to the backend), just one click away.
function SeedControl({ lastSeed }: { lastSeed: number | null }) {
  const { settings, updateSettings } = useAppSettings()
  const locked = settings.seedLocked
  const [draft, setDraft] = useState(String(settings.lockedSeed))
  useEffect(() => { setDraft(String(settings.lockedSeed)) }, [settings.lockedSeed])

  const commitDraft = () => {
    const n = parseInt(draft, 10)
    if (Number.isFinite(n)) updateSettings({ lockedSeed: Math.max(0, Math.min(MAX_SEED, n)) })
    else setDraft(String(settings.lockedSeed))
  }

  return (
    <div className="ml-2 flex items-center gap-0.5">
      <button
        type="button"
        onClick={() => updateSettings({ seedLocked: !locked })}
        title={locked ? 'Seed locked — click for a random seed each generation' : 'Random seed each generation — click to lock the seed'}
        className={`flex items-center gap-1 px-2 py-1.5 rounded-md text-[11px] font-semibold transition-colors ${
          locked ? 'bg-emerald-600 hover:bg-emerald-500 text-white' : 'bg-zinc-800 hover:bg-zinc-700 text-zinc-300'
        }`}
      >
        {locked ? <Lock className="h-3.5 w-3.5" /> : <Dices className="h-3.5 w-3.5" />}
        {locked ? 'Locked' : 'Random'}
      </button>
      {locked && (
        <>
          <input
            type="text"
            inputMode="numeric"
            value={draft}
            onChange={(e) => setDraft(e.target.value.replace(/[^0-9]/g, ''))}
            onBlur={commitDraft}
            onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }}
            title="Locked seed"
            className="w-[88px] px-1.5 py-1 rounded-md bg-zinc-900 border border-zinc-700 text-zinc-200 text-[11px] font-mono focus:outline-none focus:border-emerald-500"
          />
          <button
            type="button"
            onClick={() => updateSettings({ lockedSeed: randomSeed() })}
            title="Roll a new random seed (stays locked)"
            className="p-1.5 rounded-md text-zinc-400 hover:text-white hover:bg-zinc-700 transition-colors"
          >
            <Dices className="h-3.5 w-3.5" />
          </button>
        </>
      )}
      {lastSeed != null && !(locked && settings.lockedSeed === lastSeed) && (
        <button
          type="button"
          onClick={() => updateSettings({ seedLocked: true, lockedSeed: lastSeed })}
          title={`Lock the last generation's seed (${lastSeed})`}
          className="flex items-center gap-1 p-1.5 rounded-md text-zinc-400 hover:text-white hover:bg-zinc-700 transition-colors text-[11px]"
        >
          <History className="h-3.5 w-3.5" />
          <span className="font-mono">{lastSeed}</span>
        </button>
      )}
    </div>
  )
}

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
  canUseMultiKeyframe,
  canUseIcLora,
  canUseRetake,
  canUseExtend,
  canUseUserLoras,
  prompt,
  onPromptChange,
  onClearPrompt,
  onGenerate,
  onStop,
  isGenerating,
  isCancelling,
  inputImage,
  onInputImageChange,
  inputLastImage,
  onInputLastImageChange,
  inputAudio,
  onInputAudioChange,
  keyframes,
  onKeyframesChange,
  multiKeyframeMaxCount,
  playheadFrame,
  onPlayheadChange,
  onDragFrameChange,
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
  retakeExtendModel,
  onRetakeExtendModelChange,
  icLoraControls,
  promptOptional,
  isLocalMode,
  imageUsesFalApi,
  availableLoras,
  selectedLoras,
  onSelectedLorasChange,
  loraDisplayNames,
  loraCatalogIdsByPath,
  enhanceAvailableForMode,
  canEnhancePrompt,
  enhanceBlockedByMissingGeminiKey,
  isEnhancingPrompt,
  enhancePromptError,
  onEnhancePrompt,
  enhanceProvider,
  onEnhanceProviderChange,
  canUndoPrompt,
  onUndoPrompt,
  canRedoPrompt,
  onRedoPrompt,
  allowUltrawideVideo,
  lastSeed,
}: {
  mode: GenSpaceMode
  onModeChange: (mode: GenSpaceMode) => void
  canUseMultiKeyframe: boolean
  canUseIcLora: boolean
  canUseRetake: boolean
  canUseExtend: boolean
  canUseUserLoras: boolean
  prompt: string
  onPromptChange: (prompt: string) => void
  onClearPrompt: () => void
  onGenerate: () => void
  onStop?: () => void
  isGenerating: boolean
  isCancelling?: boolean
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
  // Retake/extend, API mode: ltxv-api /v1/retake and /v2/extend accept
  // ltx-2-pro / ltx-2-3-pro (Desktop pipeline "pro").
  retakeExtendModel?: RetakeExtendModel
  onRetakeExtendModelChange?: (model: RetakeExtendModel) => void
  inputImage: string | null
  onInputImageChange: (path: string | null) => void
  inputLastImage: string | null
  onInputLastImageChange: (path: string | null) => void
  inputAudio: string | null
  onInputAudioChange: (path: string | null) => void
  keyframes: readonly KeyframeItem[]
  onKeyframesChange: (keyframes: KeyframeItem[]) => void
  multiKeyframeMaxCount: number
  playheadFrame: number
  onPlayheadChange: (frameIndex: number) => void
  onDragFrameChange: (drag: DraggedFrame | null) => void
  settings: {
    model: string
    duration: number | null
    videoResolution: string
    fps: number
    aspectRatio: string
    imageResolution: string
    imageModel: string
    variations: number
    audio?: boolean
    imageEditStrength?: number
    imageVariation?: number
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
  enhanceAvailableForMode?: boolean
  canEnhancePrompt?: boolean
  enhanceBlockedByMissingGeminiKey?: boolean
  isEnhancingPrompt?: boolean
  enhancePromptError?: string | null
  onEnhancePrompt?: () => void
  enhanceProvider?: 'local' | 'api'
  onEnhanceProviderChange?: (provider: 'local' | 'api') => void
  canUndoPrompt?: boolean
  onUndoPrompt?: () => void
  canRedoPrompt?: boolean
  onRedoPrompt?: () => void
  allowUltrawideVideo?: boolean
  lastSeed?: number | null
}) {
  const enhanceDisabled = (!canEnhancePrompt && !enhanceBlockedByMissingGeminiKey) || !!isEnhancingPrompt
  const inputRef = useRef<HTMLInputElement>(null)
  const lastFrameInputRef = useRef<HTMLInputElement>(null)
  const audioInputRef = useRef<HTMLInputElement>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [isLastDragOver, setIsLastDragOver] = useState(false)
  const [isAudioDragOver, setIsAudioDragOver] = useState(false)
  const [promptBoxHeight, setPromptBoxHeight] = useState(70)
  const isRetake = mode === 'retake'
  const isExtend = mode === 'extend'
  const isIcLora = mode === 'ic-lora'
  const isEditingImage = mode === 'image' && !!inputImage
  const availableModeValues = modeOptionValues({
    canUseMultiKeyframe,
    canUseRetake,
    canUseExtend,
    canUseIcLora,
  })

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

  // Unused while RETAKE_EXTEND_MODELS is only "pro".
  const showRetakeExtendModel = !isLocalMode && (isRetake || isExtend) && RETAKE_EXTEND_MODELS.length > 1
  const retakeExtendModelLabel = (value: RetakeExtendModel) =>
    formatPipelineDisplayName(value) ?? value
  const modelControl = showRetakeExtendModel ? (
    <SettingsDropdown
      title="MODEL"
      value={retakeExtendModel ?? 'pro'}
      onChange={(v) => onRetakeExtendModelChange?.(v as RetakeExtendModel)}
      options={RETAKE_EXTEND_MODELS.map((value) => ({
        value,
        label: retakeExtendModelLabel(value),
      }))}
      trigger={
        <>
          <Sparkles className="h-3.5 w-3.5" />
          <span>{retakeExtendModelLabel(retakeExtendModel ?? 'pro')}</span>
          <ChevronUp className="h-3 w-3 text-zinc-500" />
        </>
      }
    />
  ) : null
  const resolvedVideoOptions = mode === 'video' || mode === 'multi-keyframe'
    ? resolveVideoGenerationOptions({
        settings,
        modelSpecs: videoModelSpecs,
        hasAudio: genSpaceUsesAudioInput(mode) && Boolean(inputAudio),
        minimumDuration: isLocalMode ? undefined : GENSPACE_MIN_SELECTABLE_DURATION_S,
      })
    : null
  const showVideoFpsControl = Boolean(
    resolvedVideoOptions
    && resolvedVideoOptions.hasCompatibleOptions
    && resolvedVideoOptions.fpsOptions.length > 1,
  )
  const showAutoDurationOption = Boolean(
    resolvedVideoOptions
    && autoDurationOptionVisible(mode, resolvedVideoOptions.autoDurationAvailable),
  )
  const selectedDuration = showAutoDurationOption && resolvedVideoOptions?.selectedDuration === null
    ? null
    : resolvedVideoOptions?.selectedDuration
      ?? resolvedVideoOptions?.durationOptions[0]
      ?? settings.duration

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    // Image dragged from the Prompt Manager Pro library (a dataURL): persist to
    // a file, then use it as the input image.
    const gpmData = e.dataTransfer.getData(GPM_IMAGE_DND_TYPE)
    if (gpmData) {
      const { name, dataUrl } = JSON.parse(gpmData) as GpmDndImage
      void saveDataUrlToTempFile(dataUrl, name).then(onInputImageChange).catch(() => {})
      return
    }

    // Image dragged from the Downloads Browser — already a real file on disk.
    const dlData = e.dataTransfer.getData(FILE_DND)
    if (dlData) {
      const f = JSON.parse(dlData) as LibFile
      if (!f.isVideo) onInputImageChange(f.path)
      return
    }

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

  const handleLastDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsLastDragOver(false)

    // Mirror handleDrop's full source handling so the last-frame chip accepts the
    // same drags the first-frame chip does: Prompt Manager Pro images, Downloads
    // Browser files, gallery assets, and OS file drops. (The last-frame handler
    // used to only take gallery assets + OS files, so library drags silently no-op'd.)
    const gpmData = e.dataTransfer.getData(GPM_IMAGE_DND_TYPE)
    if (gpmData) {
      const { name, dataUrl } = JSON.parse(gpmData) as GpmDndImage
      void saveDataUrlToTempFile(dataUrl, name).then(onInputLastImageChange).catch(() => {})
      return
    }

    const dlData = e.dataTransfer.getData(FILE_DND)
    if (dlData) {
      const f = JSON.parse(dlData) as LibFile
      if (!f.isVideo) onInputLastImageChange(f.path)
      return
    }

    const assetData = e.dataTransfer.getData('asset')
    if (assetData) {
      const asset = JSON.parse(assetData) as Asset
      if (asset.type === 'image') {
        onInputLastImageChange(asset.path)
      }
      return
    }

    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('image/')) {
      const filePath = window.electronAPI?.getPathForFile(file)
      onInputLastImageChange(filePath || URL.createObjectURL(file))
    }
  }

  const handleLastFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type.startsWith('image/')) {
      const filePath = window.electronAPI?.getPathForFile(file)
      if (filePath) {
        onInputLastImageChange(filePath)
      }
    }
  }
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isGenerating && canGenerate && !isEnhancingPrompt) {
      e.preventDefault()
      onGenerate()
    }
  }

  // Custom vertical resize: grab the handle above the prompt box and drag UP to
  // grow it (native textarea resize grows downward from the bottom-right grip,
  // which felt backwards). Dragging up = taller, dragging down = shorter.
  const MIN_PROMPT_HEIGHT = 70
  const MAX_PROMPT_HEIGHT = Math.round(window.innerHeight * 0.6)
  const [copiedPromptBox, setCopiedPromptBox] = useState(false)
  const handleResizeStart = useCallback((e: React.PointerEvent) => {
    e.preventDefault()
    const startY = e.clientY
    const startHeight = promptBoxHeight
    const onMove = (ev: PointerEvent) => {
      // Dragging up (clientY decreases) increases height.
      const next = startHeight + (startY - ev.clientY)
      setPromptBoxHeight(Math.max(MIN_PROMPT_HEIGHT, Math.min(MAX_PROMPT_HEIGHT, next)))
    }
    const onUp = () => {
      document.removeEventListener('pointermove', onMove)
      document.removeEventListener('pointerup', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    document.addEventListener('pointermove', onMove)
    document.addEventListener('pointerup', onUp)
    document.body.style.cursor = 'ns-resize'
    document.body.style.userSelect = 'none'
  }, [promptBoxHeight, MAX_PROMPT_HEIGHT])

  const showStop = Boolean(isGenerating && onStop)
  const stopDisabled = Boolean(isCancelling)
  const generateDisabled = isGenerating || !canGenerate || isEnhancingPrompt

  return (
    <div className="flex flex-col bg-zinc-900 border border-zinc-800 rounded-2xl overflow-visible">
      {mode === 'multi-keyframe' && (
        <MultiKeyframePanel
          keyframes={keyframes}
          duration={resolvedVideoOptions?.selectedDuration ?? settings.duration}
          fps={resolvedVideoOptions?.selectedFps ?? settings.fps}
          maxCount={multiKeyframeMaxCount}
          playheadFrame={playheadFrame}
          onPlayheadChange={onPlayheadChange}
          onChange={onKeyframesChange}
          onDragFrameChange={onDragFrameChange}
        />
      )}
      {/* Top row: Image ref | Prompt | Generate */}
      <div className="flex items-start">
        {/* Input image drop zone — video mode (I2V) or image mode (edit source) */}
        {(mode === 'video' || mode === 'image') && !isRetake && !isIcLora && (
          <div
            className={`relative w-32 aspect-video mx-2 mt-2 self-start rounded-lg border-2 border-dashed transition-colors flex items-center justify-center flex-shrink-0 cursor-pointer ${
              isDragOver ? 'border-blue-500 bg-blue-500/10' : 'border-zinc-700 hover:border-zinc-500'
            }`}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
          >
            {inputImage ? (
              <>
                <img src={pathToFileUrl(inputImage)} alt="" className="w-full h-full object-contain rounded-md" />
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    onInputImageChange(null)
                    onInputLastImageChange(null)
                  }}
                  className="absolute -top-1 -right-1 p-0.5 rounded-full bg-zinc-800 text-zinc-400 hover:text-white z-10"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </>
            ) : (
              <Image className="h-10 w-10 text-zinc-500" />
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

        {shouldShowLastFrameChip({
          mode,
          hasFirstFrame: Boolean(inputImage),
          duration: settings.duration,
        }) && !isRetake && !isIcLora && (
          <div
            className={`relative w-32 aspect-video mt-2 mr-2 rounded-lg border-2 border-dashed transition-colors flex items-center justify-center flex-shrink-0 cursor-pointer ${
              isLastDragOver ? 'border-blue-500 bg-blue-500/10' : 'border-zinc-700 hover:border-zinc-500'
            }`}
            title="Last frame"
            onDragOver={(e) => { e.preventDefault(); setIsLastDragOver(true) }}
            onDragLeave={() => setIsLastDragOver(false)}
            onDrop={handleLastDrop}
            onClick={() => lastFrameInputRef.current?.click()}
          >
            {inputLastImage ? (
              <>
                <img src={pathToFileUrl(inputLastImage)} alt="" className="w-full h-full object-cover rounded-md" />
                <button
                  onClick={(e) => { e.stopPropagation(); onInputLastImageChange(null) }}
                  className="absolute -top-1 -right-1 p-0.5 rounded-full bg-zinc-800 text-zinc-400 hover:text-white z-10"
                >
                  <X className="h-3 w-3" />
                </button>
              </>
            ) : (
              <Image className="h-8 w-8 text-zinc-500" />
            )}
            <input
              ref={lastFrameInputRef}
              type="file"
              accept="image/*"
              onChange={handleLastFileSelect}
              className="hidden"
            />
          </div>
        )}

        {/* Audio drop zone — A2V is mutually exclusive with multi-keyframe */}
        {genSpaceUsesAudioInput(mode) && !isRetake && !isIcLora && (
          <div
            className={`relative w-20 h-20 mt-2 self-start rounded-lg border-2 border-dashed transition-colors flex items-center justify-center flex-shrink-0 cursor-pointer ${
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
                <Music className="h-10 w-10 text-emerald-400" />
                <button
                  onClick={(e) => { e.stopPropagation(); onInputAudioChange(null) }}
                  className="absolute -top-1 -right-1 p-1 rounded-full bg-zinc-800 text-zinc-400 hover:text-white z-10"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </>
            ) : (
              <Music className="h-10 w-10 text-zinc-500" />
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

        {/* Prompt input - fills remaining width; drag the handle above it to resize height */}
        <div className="flex-1 min-w-0 py-1">
          {/* Resize handle: thick, easy-to-grab bar; drag UP to grow the box. */}
          <div className="relative">
            <div
              onPointerDown={handleResizeStart}
              title="Drag up to enlarge the prompt box"
              className="group flex h-3.5 w-full cursor-ns-resize items-center justify-center touch-none"
            >
              <div className="h-0 w-0 border-l-[8px] border-r-[8px] border-b-[10px] border-l-transparent border-r-transparent border-b-accent/70 group-hover:border-b-accent transition-colors" />
            </div>
            {prompt.trim() && (
              <button
                type="button"
                onClick={() => {
                  navigator.clipboard.writeText(prompt)
                  setCopiedPromptBox(true)
                  setTimeout(() => setCopiedPromptBox(false), 2000)
                }}
                className="absolute right-0 top-0 p-0.5 rounded text-zinc-500 hover:text-zinc-200 hover:bg-zinc-700 transition-colors"
                title="Copy prompt"
              >
                {copiedPromptBox ? <Check className="h-3.5 w-3.5 text-green-400" /> : <Copy className="h-3.5 w-3.5" />}
              </button>
            )}
          </div>
          <textarea
            value={prompt}
            onChange={(e) => onPromptChange(e.target.value)}
            onKeyDown={handleKeyDown}
            spellCheck
            style={{ height: `${promptBoxHeight}px` }}
            onDragOver={(e) => {
              if (e.dataTransfer.types.includes(GPM_IMAGE_DND_TYPE) || e.dataTransfer.types.includes(FILE_DND) || e.dataTransfer.types.includes('asset')) e.preventDefault()
            }}
            onDrop={(e) => {
              if (e.dataTransfer.types.includes(GPM_IMAGE_DND_TYPE) || e.dataTransfer.types.includes(FILE_DND) || e.dataTransfer.types.includes('asset')) handleDrop(e)
            }}
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
            className="prompt-scroll w-full bg-transparent text-white text-sm placeholder:text-zinc-500 focus:outline-none px-2 py-2 resize-none overflow-y-auto leading-5"
          />
        </div>

      </div>
      
      {/* Bottom row: Mode selector + Settings. Wraps (flex-wrap) so a narrow panel
          pushes overflowing controls onto a second line INSIDE the box, instead of
          spilling the Generate button past the rounded border. */}
      <div className="flex flex-wrap flex-shrink-0 items-center gap-0.5 gap-y-1.5 px-1.5 py-1.5 border-t border-zinc-800/60 text-xs text-zinc-400">
        {/* Mode dropdown */}
        <SettingsDropdown
          title="MODE"
          value={mode}
          onChange={(v) => onModeChange(v as GenSpaceMode)}
          options={[
            { value: 'image', label: 'Generate Images', icon: <Image className="h-4 w-4" /> },
            { value: 'video', label: 'Generate Videos', icon: <Video className="h-4 w-4" /> },
            { value: 'multi-keyframe', label: 'Generate Multi Keyframes Videos', icon: <Rows3 className="h-4 w-4" /> },
            { value: 'retake', label: 'Retake', icon: <Scissors className="h-4 w-4" /> },
            { value: 'extend', label: 'Extend', icon: <MoveHorizontal className="h-4 w-4" /> },
            { value: 'ic-lora', label: 'IC-LoRA', icon: <Sparkles className="h-4 w-4" /> },
          ].filter((option) => availableModeValues.includes(option.value as GenSpaceMode))}
          trigger={
            <>
              {mode === 'image' ? <Image className="h-3.5 w-3.5" /> : mode === 'multi-keyframe' ? <Rows3 className="h-3.5 w-3.5" /> : mode === 'retake' ? <Scissors className="h-3.5 w-3.5" /> : mode === 'extend' ? <MoveHorizontal className="h-3.5 w-3.5" /> : mode === 'ic-lora' ? <Sparkles className="h-3.5 w-3.5" /> : <Video className="h-3.5 w-3.5" />}
              <span className="text-zinc-300 font-medium">{mode === 'image' ? 'Image' : mode === 'multi-keyframe' ? 'Multi Keyframes' : mode === 'retake' ? 'Retake' : mode === 'extend' ? 'Extend' : mode === 'ic-lora' ? 'IC-LoRA' : 'Video'}</span>
              <ChevronUp className="h-3 w-3 text-zinc-500" />
            </>
          }
        />

        {/* Clear prompt — parked next to the mode selector, close to the box. */}
        <button
          onClick={onClearPrompt}
          disabled={!prompt}
          title="Clear the prompt box"
          className="ml-8 flex items-center gap-1 px-2.5 py-1.5 rounded-md text-[11px] font-semibold bg-red-600 hover:bg-red-500 text-white transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Eraser className="h-3.5 w-3.5" />
          Clear
        </button>

        <SeedControl lastSeed={lastSeed ?? null} />

        <div className="flex-1" />

        {isRetake ? (
          <>
            {modelControl}
            {modelControl && resolutionControl && (
              <div className="w-px h-4 bg-zinc-700 mx-0.5" />
            )}
            {resolutionControl}
            {/* Always available: in API mode modelControl used to replace this hint, so a
                <2s selection disabled submit with no explanation. */}
            <div className="text-[10px] text-zinc-500 pr-2">Trim in the panel above, then retake</div>
          </>
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
            {modelControl && (
              <>
                <div className="w-px h-4 bg-zinc-700 mx-0.5" />
                {modelControl}
              </>
            )}
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
                {/* Model dropdown (FORK: Krea 2 / Z-Image selection) */}
                <SettingsDropdown
                  title="MODEL"
                  value={settings.imageModel}
                  onChange={(v) => onSettingsChange({ ...settings, imageModel: v })}
                  options={[
                    { value: 'z-image-turbo', label: 'Z-Image Turbo' },
                    ...(KREA2_ENABLED ? [{ value: 'krea-2-turbo', label: 'Krea 2 Turbo' }] : []),
                  ]}
                  trigger={
                    <>
                      {allowedImageModel(settings.imageModel) === 'krea-2-turbo' ? <Sparkles className="h-3.5 w-3.5" /> : <ZitIcon className="h-3.5 w-3.5" />}
                      <span className="text-zinc-300 font-medium">
                        {allowedImageModel(settings.imageModel) === 'krea-2-turbo' ? 'Krea 2 Turbo' : 'Z-Image Turbo'}{imageUsesFalApi ? ' (API)' : ''}
                      </span>
                      <ChevronUp className="h-3 w-3 text-zinc-500" />
                    </>
                  }
                />

                {/* Variation boost: the turbo image models barely change composition across seeds,
                    so this perturbs the prompt embeddings on the first step(s) (local only). */}
                {!imageUsesFalApi && (
                  <div
                    className="flex items-center gap-1.5 px-2 text-[10px] text-zinc-400"
                    title="Variation boost — more composition/pose/camera variety between seeds. 0 = off. Same seed + same value reproduces the same image. Above ~70% it gets looser with the prompt."
                  >
                    <Shuffle className="h-3.5 w-3.5" />
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={settings.imageVariation ?? 0}
                      onChange={(e) => onSettingsChange({ ...settings, imageVariation: parseFloat(e.target.value) })}
                      onDoubleClick={() => onSettingsChange({ ...settings, imageVariation: 0 })}
                      className="w-16 accent-white"
                    />
                    <span className="w-7 text-right">{Math.round((settings.imageVariation ?? 0) * 100)}%</span>
                  </div>
                )}

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

                {/* Aspect ratio dropdown (FORK: keeps 21:9) */}
                <SettingsDropdown
                  title="RATIO"
                  value={settings.aspectRatio}
                  onChange={(v) => onSettingsChange({ ...settings, aspectRatio: v })}
                  options={[
                    { value: '16:9', label: '16:9' },
                    { value: '1:1', label: '1:1' },
                    { value: '9:16', label: '9:16' },
                    { value: '21:9', label: '21:9' },
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
                  value={selectedDuration === null ? 'auto' : String(selectedDuration)}
                  onChange={(v) => onSettingsChange({
                    ...settings,
                    duration: v === 'auto' ? null : parseInt(v),
                  })}
                  options={[
                    ...(showAutoDurationOption
                      ? [{ value: 'auto', label: 'Auto' }]
                      : []),
                    ...resolvedVideoOptions.durationOptions.map((value) => ({
                      value: String(value),
                      label: `${value} Sec`,
                    })),
                  ]}
                  trigger={
                    <>
                      <Clock className="h-3.5 w-3.5" />
                      <span>
                        {selectedDuration === null ? 'Auto' : `${selectedDuration}s`}
                      </span>
                    </>
                  }
                />

                <SettingsDropdown
                  title="RESOLUTION"
                  value={resolvedVideoOptions.selectedResolution ?? settings.videoResolution}
                  onChange={(v) => onSettingsChange({ ...settings, videoResolution: v })}
                  options={resolvedVideoOptions.resolutionOptions.map((value) => ({
                    value,
                    label: videoGenerationResolutionLabel(value),
                  }))}
                  trigger={
                    <>
                      <Monitor className="h-3.5 w-3.5" />
                      <span>
                        {videoGenerationResolutionLabel(
                          resolvedVideoOptions.selectedResolution ?? settings.videoResolution,
                        ).replace(/p$/, '')}
                      </span>
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
                    // 21:9 ultrawide is local-only (the LTX cloud API rejects it).
                    ...(allowUltrawideVideo ? [{ value: '21:9', label: '21:9' }] : []),
                  ]}
                  trigger={
                    <>
                      <AspectIcon className="h-3.5 w-3.5" />
                      <span>{settings.aspectRatio}</span>
                    </>
                  }
                />

                {mode === 'video' && isLocalMode && canUseUserLoras && availableLoras && availableLoras.length > 0 && (
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
        
        {/* Catalog-aware prompt enhancer — video/multi-keyframe/IC-LoRA (local-generation-only)
            or image (generation/editing, any backend). Runs either the local Gemma text encoder
            or, if available, Gemini's hosted API. */}
        {enhanceAvailableForMode && (
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
                disabled={enhanceDisabled}
                title={enhancePromptError ?? 'Enhance prompt'}
                className={`flex items-center gap-1 px-2 py-1.5 text-xs font-medium ${
                  onEnhanceProviderChange ? 'rounded-l-md' : 'rounded-md'
                } ${
                  enhanceDisabled
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

        {/* Generate / Stop button */}
        <button
          onClick={showStop ? onStop : onGenerate}
          disabled={showStop ? stopDisabled : generateDisabled}
          className={`flex items-center gap-1.5 ml-auto px-3 py-1.5 rounded-md text-xs font-medium transition-all flex-shrink-0 ${
            showStop
              ? stopDisabled
                ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                : 'bg-red-600 text-white hover:bg-red-500'
              : generateDisabled
                ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                : 'bg-white text-black hover:bg-zinc-200'
          }`}
        >
          {showStop ? (
            <>
              <Square className="h-3.5 w-3.5 fill-current" />
              {isCancelling ? 'Stopping...' : 'Stop'}
            </>
          ) : (
            <>
              <span className={isGenerating ? 'animate-pulse' : ''}>{buttonIcon}</span>
              {buttonLabel}
            </>
          )}
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

// Auto-fill with a fixed min thumbnail width (rather than viewport-breakpoint
// grid-cols) so thumbnails stay a constant size when side panels open/close —
// only the column count adapts to the available width.
const gallerySizeClasses: Record<GallerySize, string> = {
  small: 'grid-cols-[repeat(auto-fill,minmax(200px,1fr))]',
  medium: 'grid-cols-[repeat(auto-fill,minmax(280px,1fr))]',
  large: 'grid-cols-[repeat(auto-fill,minmax(380px,1fr))]',
}

// Map a source clip's geometry to Gen Space's discrete video settings so a
// "Continue as new shot" continuation renders at a matching size/fps — the
// separate clips must cut together cleanly on the timeline.
function continuationAspectRatio(width: number, height: number): string {
  const r = width / height
  const opts: Array<[string, number]> = [['16:9', 16 / 9], ['9:16', 9 / 16], ['21:9', 21 / 9]]
  return opts.reduce((best, o) => (Math.abs(o[1] - r) < Math.abs(best[1] - r) ? o : best))[0]
}
// Fallback tier from a clip's short side, for a source with no stored resolution.
// Thresholds sit at the midpoints between tiers' actual short-side outputs (540p ~512-576,
// 720p ~704-768, 1080p ~1056+) so a 540p 16:9 clip (576) buckets as 540p, not 720p.
function continuationResolution(shortSide: number): string {
  if (shortSide <= 640) return '540p'
  if (shortSide <= 912) return '720p'
  return '1080p'
}

const DEFAULT_VIDEO_SETTINGS = {
  model: 'fast',
  duration: 5 as number | null,
  videoResolution: '540p',
  fps: 24,
  aspectRatio: '16:9',
  imageResolution: '1080p',
  imageModel: 'z-image-turbo',
  variations: 1,
  audio: true,
  imageEditStrength: 0.6,
  imageVariation: 0,
}

export function GenSpace() {
  const {
    activeProject,
    addAsset,
    addTakeToAsset,
    updateAsset,
    deleteAsset,
    toggleFavorite,
    createBin,
    deleteBin,
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
    genSpacePromptInjection,
    setGenSpacePromptInjection,
    genSpaceInputImagePath,
    setGenSpaceInputImagePath,
    genSpacePromptClearNonce,
    clearGenSpacePrompt,
  } = useProjects()
  const currentProjectId = activeProject?.id ?? null
  const { shouldVideoGenerateWithLtxApi, shouldImageGenerateWithFalApi, forceApiGenerations, settings: appSettings, updateSettings: updateAppSettings } = useAppSettings()
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
  const [inputLastImage, setInputLastImage] = useState<string | null>(null)
  const [inputAudio, setInputAudio] = useState<string | null>(null)
  const [keyframes, setKeyframes] = useState<KeyframeItem[]>([])
  const [playheadFrame, setPlayheadFrame] = useState(0)
  const [, setDragFrame] = useState<DraggedFrame | null>(null)
  const [localError, setLocalError] = useState<GenerationError | null>(null)
  const [selectedAsset, setSelectedAsset] = useState<Asset | null>(null)
  const enlargedVideoRef = useRef<HTMLVideoElement | null>(null)
  const [copiedPrompt, setCopiedPrompt] = useState(false)
  // Live render stopwatch: elapsedMs ticks while generating and freezes on
  // completion; generationStartRef feeds the render time stored on the asset.
  const [elapsedMs, setElapsedMs] = useState(0)
  const generationStartRef = useRef<number | null>(null)
  const { onContextMenu: onVideoSaveContextMenu, menu: videoSaveMenu } = useVideoSaveMenu()
  const { onContextMenu: onImageSaveContextMenu, menu: imageSaveMenu } = useImageSaveMenu()
  const [showFavorites, setShowFavorites] = useState(false)
  // null = default view (untagged only); ALL_BINS_FILTER = every asset; otherwise a specific binId.
  const [binFilter, setBinFilter] = useState<string | null>(null)
  // Electron's renderer doesn't implement window.prompt(), so new tag/folder
  // names go through this small modal instead. '__bar__' = creating from the
  // chip bar (no asset assignment); an assetId = creating + assigning to it.
  // '__bar__' = new tag from the chip bar (no assignment); '__selection__' =
  // new tag applied to the current multi-selection; an assetId = assign to it.
  const [creatingTagFor, setCreatingTagFor] = useState<'__bar__' | '__selection__' | string | null>(null)
  const [newTagName, setNewTagName] = useState('')
  const [gallerySize, setGallerySize] = useState<GallerySize>('medium')
  const [showSizeMenu, setShowSizeMenu] = useState(false)
  const sizeMenuRef = useRef<HTMLDivElement>(null)
  // Multi-select: bulk-tag/delete many assets at once. Drag a marquee over the
  // grid, or click cards, to build a selection, then tag/delete them together.
  const [selectMode, setSelectMode] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set())
  const [marquee, setMarquee] = useState<{ left: number; top: number; width: number; height: number } | null>(null)
  const gridScrollRef = useRef<HTMLDivElement>(null)
  const marqueeStartRef = useRef<{ x: number; y: number } | null>(null)
  const marqueeBaseRef = useRef<Set<string>>(new Set())
  const marqueeDraggedRef = useRef(false)
  const suppressCardClickRef = useRef(false)
  const persistedVideoKeyRef = useRef<string | null>(null)
  const retakeSubmissionRef = useRef<{
    prompt: string
    model: RetakeExtendModel
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
  // Click-time t2v/i2v/a2v/image snapshot. The live picker can change while the job
  // runs; the completion effects must tag the asset with what was actually submitted.
  // Survives a refresh via the recovery marker restore below — the ref itself does not.
  const generateSubmissionRef = useRef<{
    kind: 'video' | 'image'
    prompt: string
    settings: GenerationSettings
    modelLabel?: string
    inputImageUrl: string | null
    inputLastImageUrl: string | null
    inputAudioUrl: string | null
    keyframes?: KeyframeItem[]
  } | null>(null)
  const [settings, setSettings] = useState(() => ({ ...DEFAULT_VIDEO_SETTINGS }))
  const continuationPendingRef = useRef(false)
  const continuationProcessedRef = useRef<string | null>(null)
  // The seed frame (source clip's last frame) for the pending continuation, used to
  // color-match the trimmed clip back to the source and cancel the i2v VAE's
  // systematic darkening (see continuation-color-shift). Captured here rather than
  // read from inputImage at trim time, which the user may have cleared/changed.
  const continuationSeedRef = useRef<string | null>(null)
  const previousTimelineSettingsRef = useRef({
    duration: settings.duration,
    fps: settings.fps,
  })
  // Memoized (fork perf): rebuilding every render defeated downstream gallery memos.
  const videoModelSpecs = useMemo(
    () => getVideoGenerationModelSpecs(videoGenerationModelSpecsResponse, {
      useApiSpecs: shouldVideoGenerateWithLtxApi,
    }),
    [videoGenerationModelSpecsResponse, shouldVideoGenerateWithLtxApi],
  )
  const videoSettingsMessage = isLoadingVideoGenerationModelSpecs
    ? 'Loading generation settings...'
    : videoGenerationModelSpecsErrorMessage
      ? `Could not load generation settings: ${videoGenerationModelSpecsErrorMessage}`
      : null
  const sanitizeVideoSettings = useCallback(
    (
      next: typeof settings,
      durationSelection: 'preserve' | 'smallest_valid' = 'preserve',
    ) => {
      if ((mode !== 'video' && mode !== 'multi-keyframe') || videoModelSpecs.length === 0) return next
      return sanitizeVideoGenerationSettings(next, videoModelSpecs, {
        hasAudio: genSpaceUsesAudioInput(mode) && Boolean(inputAudio),
        // FORK: 21:9 is only valid on the local pipeline; drop it in API mode.
        allowedAspectRatios: shouldVideoGenerateWithLtxApi
          ? ['16:9', '9:16']
          : ['16:9', '9:16', '21:9'],
        minimumDuration: shouldVideoGenerateWithLtxApi ? GENSPACE_MIN_SELECTABLE_DURATION_S : undefined,
        durationSelection,
      }) ?? next
    },
    [inputAudio, mode, shouldVideoGenerateWithLtxApi, videoModelSpecs],
  )
  
  const {
    generate,
    generateImage,
    isGenerating,
    progress,
    statusMessage,
    videoPath,
    imagePaths,
    seed: generatedSeed,
    error,
    reset,
    resumeIfRunning,
    cancel,
  } = useGeneration()

  // generatedSeed clears on reset(); remember the most recent one for the prompt bar's
  // "lock last seed" button.
  const [lastSeed, setLastSeed] = useState<number | null>(null)
  useEffect(() => { if (generatedSeed != null) setLastSeed(generatedSeed) }, [generatedSeed])

  // Locally installed LoRAs are only usable in local generation mode.
  const isLocalMode = !shouldVideoGenerateWithLtxApi
  const localCaps = getLocalOfferingCapabilities(videoGenerationModelSpecsResponse)
  // Retake/Extend always request RETAKE_EXTEND_MODELS (currently "pro" / 2.3 Pro),
  // not the t2v/i2v picker in settings.model.
  const apiCaps = getApiOfferingCapabilities(
    videoGenerationModelSpecsResponse,
    RETAKE_EXTEND_MODELS[0],
  )
  const canUseUserLoras = isLocalMode && Boolean(localCaps?.user_loras)
  const multiKeyframeMaxCount =
    localCaps?.multi_keyframe_max_count ?? LOCAL_MULTI_KEYFRAME_MAX_COUNT
  const canUseIcLora = !forceApiGenerations && Boolean(localCaps?.ic_lora)
  const canUseRetake = isLocalMode ? Boolean(localCaps?.retake) : Boolean(apiCaps?.retake)
  const canUseExtend = isLocalMode ? Boolean(localCaps?.extend) : Boolean(apiCaps?.extend)
  // Enhance itself is independent of the video-generation backend — the backend enhance
  // endpoint only cares about the enhancer provider (local Gemma vs. Gemini), not whether video
  // generation runs locally or via the LTX API. If no catalog LoRA is selected (e.g. because the
  // LoRA picker is local-only), it just falls back to a generic rewrite. "retake"/"extend" have
  // no prompt input, so they're excluded. Multi-keyframe uses the same video enhance path, driven
  // by timeline stills (and optional prompt text).
  const enhanceAvailableForMode = isEnhanceAvailableForMode(mode)
  // The prompt enhancer can run the local Gemma text encoder OR Gemini's hosted API — this hook
  // tracks which of those is actually available (not just which the user prefers) and picks
  // whichever provider Enhance should use. Refetched whenever the user is in a mode the button
  // could appear in, so downloading the checkpoint from Settings and coming back here picks it up.
  const {
    hasGeminiApiKey,
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
  const { enableMultipleKeyframesVideos, advancedIcLoraControls } = useDevFlags().flags
  const canUseMultiKeyframe = canUseMultiKeyframeMode({
    isLocalMode,
    localCaps,
    enableMultipleKeyframesVideos,
  })
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

  // Default the IC-LoRA picker to Ingredients (vs the built-in Canny Edges) each time IC-LoRA
  // mode is entered from the mode dropdown — saves a click. Wrinkles this handles:
  //  - The catalog (`icLoras`) is only fetched once the mode is active, so we can't select
  //    synchronously at switch time; the wrapper flags intent and the effect applies it once
  //    the list loads.
  //  - The ICLoraPanel remounts on every entry and its resetKey effect forces conditioning
  //    back to Canny (clearing the selection + settings). Because this parent effect runs
  //    after that child effect, re-selecting Ingredients here wins — and the flag is set on
  //    EVERY dropdown entry so it re-asserts each time, not just the first.
  //  - Only the *dropdown* sets the flag. The gallery's "IC-LoRA" button on a video enters
  //    ic-lora mode (canny/depth on that clip) via a different path, so it's left alone.
  const wantIcLoraDefaultRef = useRef(false)
  const handleGenSpaceModeChange = useCallback((next: GenSpaceMode) => {
    if (next === 'ic-lora' && mode !== 'ic-lora') wantIcLoraDefaultRef.current = true
    setMode(next)
  }, [mode])
  useEffect(() => {
    if (mode !== 'ic-lora') { wantIcLoraDefaultRef.current = false; return }
    if (!wantIcLoraDefaultRef.current) return
    const item = icLoras.find(r => r.ic_lora.id === DEFAULT_IC_LORA_ID)
    const meta = icLoraItems.find(e => e.id === DEFAULT_IC_LORA_ID)
    if (!item || !meta?.downloaded) return // wait for the catalog to load; skip if not downloaded
    wantIcLoraDefaultRef.current = false
    // Applies the catalog default_settings (Stage 2 off, LoRA-in-S2, res ×2, strength 1.4) via
    // onSelectIcLora — no separate settings step needed here.
    selectIcLora(item)
  }, [mode, icLoras, icLoraItems, selectIcLora])
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
  const [retakeModel, setRetakeModel] = useState<RetakeExtendModel>('pro')
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
  const [extendModel, setExtendModel] = useState<RetakeExtendModel>('pro')

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
    model: RetakeExtendModel
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

  // Handle prompt text injected from Prompt Manager Pro. Append to any existing
  // prompt (on its own line) so it composes with whatever the user is writing.
  useEffect(() => {
    if (genSpacePromptInjection == null) return
    const injected = genSpacePromptInjection
    setPrompt((prev) => (prev.trim() ? `${prev.trimEnd()}\n${injected}` : injected))
    setGenSpacePromptInjection(null)
  }, [genSpacePromptInjection, setGenSpacePromptInjection])

  // Handle reference image sent from Prompt Manager Pro (keeps the prompt intact).
  useEffect(() => {
    if (!genSpaceInputImagePath) return
    setMode('video')
    setInputImage(genSpaceInputImagePath)
    setGenSpaceInputImagePath(null)
  }, [genSpaceInputImagePath, setGenSpaceInputImagePath])

  // Clear the prompt box on request from Prompt Manager Pro (skip initial mount).
  const lastClearNonce = useRef(genSpacePromptClearNonce)
  useEffect(() => {
    if (genSpacePromptClearNonce !== lastClearNonce.current) {
      lastClearNonce.current = genSpacePromptClearNonce
      setPrompt('')
    }
  }, [genSpacePromptClearNonce])

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
    // Specs start null on remount (Project unmounts GenSpace off-tab). Don't drop an
    // incoming timeline retake until we know whether the offering actually supports it.
    if (isLoadingVideoGenerationModelSpecs) return
    if (!canUseRetake) {
      setGenSpaceRetakeSource(null)
      return
    }
    setMode('retake')
    setPrompt('')
    setActiveRetakeSource(genSpaceRetakeSource)
    const sourceAsset = genSpaceRetakeSource.assetId
      ? activeProject?.assets?.find((a) => a.id === genSpaceRetakeSource.assetId)
      : undefined
    setRetakeModel(retakeExtendModelFromPipeline(sourceAsset?.generationParams?.model))
    setRetakeInitial({
      videoPath: genSpaceRetakeSource.videoPath,
      duration: genSpaceRetakeSource.duration,
    })
    setRetakePanelKey((prev) => prev + 1)
    setGenSpaceRetakeSource(null)
  }, [
    genSpaceRetakeSource,
    setGenSpaceRetakeSource,
    activeProject?.assets,
    canUseRetake,
    isLoadingVideoGenerationModelSpecs,
  ])

  useEffect(() => {
    if (!genSpaceIcLoraSource) return
    if (!canUseIcLora) {
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
  }, [genSpaceIcLoraSource, canUseIcLora, setGenSpaceIcLoraSource])

  useEffect(() => {
    if (isLoadingVideoGenerationModelSpecs) return
    const fallbackMode = fallbackGenSpaceMode(mode, {
      canUseMultiKeyframe,
      canUseIcLora,
      canUseRetake,
      canUseExtend,
    })
    if (fallbackMode !== mode) setMode(fallbackMode)
  }, [
    canUseMultiKeyframe,
    canUseIcLora,
    canUseRetake,
    canUseExtend,
    mode,
    isLoadingVideoGenerationModelSpecs,
  ])

  useEffect(() => {
    if (!canUseUserLoras && selectedLoras.length > 0) setSelectedLoras([])
  }, [canUseUserLoras, selectedLoras.length])

  useEffect(() => {
    const previous = previousTimelineSettingsRef.current
    const next = { duration: settings.duration, fps: settings.fps }

    if (previous.duration === next.duration && previous.fps === next.fps) return

    setKeyframes((current) => current.length === 0
      ? current
      : retimeKeyframesForSettings(current, previous, next))
    previousTimelineSettingsRef.current = next
  }, [settings.duration, settings.fps])

  useEffect(() => {
    if (settings.duration == null) return
    const lastFrame = lastFrameFromDuration(settings.duration, settings.fps)
    setPlayheadFrame((frame) => Math.min(frame, lastFrame))
  }, [settings.duration, settings.fps])

  useEffect(() => {
    if ((mode !== 'video' && mode !== 'multi-keyframe') || videoModelSpecs.length === 0) return
    setSettings((prev) => {
      const next = sanitizeVideoSettings(
        prev,
        mode === 'multi-keyframe' && prev.duration === null ? 'smallest_valid' : 'preserve',
      )
      return areVideoGenerationSettingsEquivalent(prev, next) ? prev : next
    })
  }, [mode, sanitizeVideoSettings, videoModelSpecs.length])

  useEffect(() => {
    if (mode !== 'video' || settings.duration == null || !inputImage) {
      setInputLastImage(null)
    }
  }, [mode, settings.duration, inputImage])

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
  // Memoized (fork perf): a fresh filter array every render defeated the gallery memos.
  const assets = useMemo(
    () => (activeProject?.assets || []).filter(a => a.generationParams),
    [activeProject?.assets],
  )
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
        setMode(ctx.genType === 'image' ? 'image' : ctx.keyframes?.length ? 'multi-keyframe' : 'video')
        setInputImage(ctx.inputImageUrl ?? null)
        setInputLastImage(ctx.inputLastImageUrl ?? null)
        setInputAudio(ctx.inputAudioUrl ?? null)
        setKeyframes(fromPersistedKeyframes(ctx.keyframes ?? []))
        setSelectedLoras(s.loras ?? [])
        // Restored stills already use this clip's frame grid (e.g. 10s last
        // frame 240). Adopt before setSettings so the retime effect does not
        // treat remount's default 5s as the previous clip and bunch them.
        previousTimelineSettingsRef.current = { duration: s.duration, fps: s.fps }
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
          imageVariation: s.imageVariation ?? prev.imageVariation,
        }))
        generateSubmissionRef.current = {
          kind: ctx.genType === 'image' ? 'image' : 'video',
          prompt: ctx.prompt,
          settings: s,
          modelLabel: ctx.modelLabel,
          inputImageUrl: ctx.inputImageUrl ?? null,
          inputLastImageUrl: ctx.inputLastImageUrl ?? null,
          inputAudioUrl: ctx.inputAudioUrl ?? null,
          keyframes: fromPersistedKeyframes(ctx.keyframes ?? []),
        }
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
  // Render stopwatch: (re)start when a generation begins, tick while running,
  // and freeze the last value when it stops (the cleanup clears the interval).
  useEffect(() => {
    if (!isGenerating) return
    generationStartRef.current = Date.now()
    setElapsedMs(0)
    const id = setInterval(() => {
      if (generationStartRef.current != null) {
        setElapsedMs(Date.now() - generationStartRef.current)
      }
    }, 200)
    return () => clearInterval(id)
  }, [isGenerating])

  useEffect(() => {
    if (!videoPath || !currentProjectId || isGenerating) return

    // Dedup is by persistedVideoKeyRef, not an assets path-match like the image effect:
    // addVisualAssetToProject copies to a fresh path so the source videoPath never appears
    // in assets. A reload can't double-import either — recovery clears the marker once it
    // reaches 'complete' (use-generation), so the sticky output is only restored once.

    // Wall-clock render time, captured before the async persist work below.
    const renderMs = generationStartRef.current != null
      ? Date.now() - generationStartRef.current
      : undefined

    const generationKey = videoPath
    if (persistedVideoKeyRef.current === generationKey) return
    persistedVideoKeyRef.current = generationKey

    const submission = generateSubmissionRef.current
    if (submission?.kind !== 'video') {
      logger.error('Video completed without a click-time submission; tagging from live picker state')
    }
    const usedPrompt = submission?.kind === 'video' ? submission.prompt : lastPrompt
    const usedSettings: GenerationSettings = submission?.kind === 'video'
      ? submission.settings
      : {
          model: settings.model as VideoGenerationPipeline,
          duration: settings.duration,
          videoResolution: settings.videoResolution,
          fps: settings.fps,
          audio: settings.audio,
          cameraMotion: 'none',
          aspectRatio: settings.aspectRatio,
          imageResolution: settings.imageResolution,
          imageAspectRatio: settings.aspectRatio ?? '16:9',
          imageSteps: 4,
          variations: settings.variations,
          imageEditStrength: settings.imageEditStrength,
        }
    const usedImage = submission?.kind === 'video' ? submission.inputImageUrl : inputImage
    const usedLastImage = submission?.kind === 'video' ? submission.inputLastImageUrl : inputLastImage
    const usedAudio = submission?.kind === 'video' ? submission.inputAudioUrl : inputAudio
    const usedKeyframes = submission?.kind === 'video' ? submission.keyframes : keyframes
    const genMode = videoGenerationModeFromInputs({
      keyframes: usedKeyframes,
      audioUrl: usedAudio,
      imageUrl: usedImage,
    })

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
          prompt: usedPrompt,
          resolution: usedSettings.videoResolution,
          duration: usedSettings.duration ?? undefined,
          renderMs,
          generationParams: {
            mode: genMode,
            prompt: usedPrompt,
            model: usedSettings.model,
            modelLabel: (submission?.kind === 'video' ? submission.modelLabel : undefined)
              ?? resolvePipelineDisplayName(videoModelSpecs, usedSettings.model)
              ?? undefined,
            seed: generatedSeed ?? undefined,
            duration: usedSettings.duration,
            resolution: usedSettings.videoResolution,
            fps: usedSettings.fps,
            audio: usedSettings.audio || false,
            cameraMotion: 'none',
            imageAspectRatio: usedSettings.aspectRatio,
            imageSteps: 4,
            inputImageUrl: usedImage || undefined,
            inputLastImageUrl: usedLastImage || undefined,
            inputAudioUrl: usedAudio || undefined,
            keyframes: usedKeyframes && usedKeyframes.length > 0
              ? toPersistedKeyframes(usedKeyframes)
              : undefined,
            loras: usedSettings.loras && usedSettings.loras.length > 0
              ? usedSettings.loras.map(l => ({ ...l, ref: toModelsDirRelativeRef(l.ref, appSettings.modelsDir) }))
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
        generateSubmissionRef.current = null
        reset()
        const nextMode = modeAfterCompletedGeneration(genMode)
        if (nextMode) setMode(nextMode)
      } catch (err) {
        persistedVideoKeyRef.current = null
        logger.error(`Failed to persist generated video asset: ${err}`)
      }
    })()
  }, [videoPath, generatedSeed, currentProjectId, isGenerating, settings, inputImage, inputLastImage, inputAudio, keyframes, lastPrompt, addAsset, reset, appSettings.modelsDir, videoModelSpecs, setMode])

  // "Continue as new shot": once the seeded continuation finishes, drop the
  // duplicate lead frame (a copy of the source's last frame the i2v conditioning
  // reproduces) and save the clean clip into the Continuations library folder,
  // ready to butt-join the source with no stutter. Independent of the persist
  // effect above; guarded by refs so it runs once per output.
  useEffect(() => {
    if (!videoPath || !continuationPendingRef.current) return
    if (continuationProcessedRef.current === videoPath) return
    continuationProcessedRef.current = videoPath
    continuationPendingRef.current = false
    const api = window.electronAPI
    if (!api) return
    const src = videoPath
    const colorMatchReference = continuationSeedRef.current ?? undefined
    void (async () => {
      try {
        await api.continuationSaveTrimmed({ videoPath: src, colorMatchReference })
      } catch (err) {
        logger.error(`Continuation trim/save failed: ${err}`)
      }
    })()
  }, [videoPath])

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
          // Takes don't carry their own model; bump the parent asset's pipeline label so a
          // 2.5 retake of a 2.3 clip doesn't keep showing the source pipeline in preview.
          // Only in API mode — local retakes don't select a MODEL and shouldn't overwrite.
          if (!isLocalMode && sourceAsset.generationParams) {
            updateAsset(currentProjectId, sourceAsset.id, {
              generationParams: {
                ...sourceAsset.generationParams,
                model: submission.model,
                modelLabel: resolvePipelineDisplayName(videoModelSpecs, submission.model) ?? undefined,
              },
            })
          }
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
            // Local mode hides the MODEL dropdown; don't persist the leftover API-mode
            // selection (e.g. 'pro' / 'pro-2.5') as a confident pipeline label.
            model: isLocalMode ? '' : submission.model,
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
  }, [retakeResult, isRetaking, currentProjectId, activeProject?.assets, activeRetakeSource, addAsset, addTakeToAsset, updateAsset, setPendingRetakeUpdate, resetRetake, isLocalMode])

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
          // Local mode hides the MODEL dropdown; don't persist a leftover API selection.
          model: isLocalMode ? '' : submission.model,
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
  }, [extendResult, isExtending, currentProjectId, addAsset, resetExtend, isLocalMode])

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

    // Wall-clock render time for this generation, mirroring the video path. A batch
    // (variations) shares one stopwatch, so every image in the batch is tagged with the
    // same total — the number the user watched tick on the Generating card.
    const renderMs = generationStartRef.current != null
      ? Date.now() - generationStartRef.current
      : undefined

    const submission = generateSubmissionRef.current
    if (submission?.kind !== 'image') {
      logger.error('Image completed without a click-time submission; tagging from live picker state')
    }
    const usedPrompt = submission?.kind === 'image' ? submission.prompt : lastPrompt
    const usedSettings: GenerationSettings = submission?.kind === 'image'
      ? submission.settings
      : {
          model: 'fast',
          duration: 5,
          videoResolution: settings.videoResolution,
          fps: 24,
          audio: false,
          cameraMotion: 'none',
          imageResolution: settings.imageResolution,
          imageAspectRatio: settings.aspectRatio ?? '16:9',
          imageSteps: 4,
        }
    const editContext = lastImageEditRef.current
    const genMode = editContext ? 'image-edit' : 'text-to-image'
    // Edits always run on Z-Image (see use-generation), whatever the picker says.
    const imageModelLabel = !editContext && usedSettings.imageModel === 'krea-2-turbo' ? 'Krea 2 Turbo' : 'Z-Image Turbo'

    ;(async () => {
      try {
        for (const [batchIndex, imgPath] of imagePaths.entries()) {
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
            prompt: usedPrompt,
            resolution: usedSettings.imageResolution,
            renderMs,
            generationParams: {
              mode: genMode,
              prompt: usedPrompt,
              model: 'fast',
              modelLabel: imageModelLabel,
              // Batch image i was generated with base seed + i.
              seed: generatedSeed != null ? generatedSeed + batchIndex : undefined,
              duration: 5,
              resolution: usedSettings.imageResolution,
              fps: 24,
              audio: false,
              cameraMotion: 'none',
              imageAspectRatio: usedSettings.imageAspectRatio || usedSettings.aspectRatio,
              imageSteps: editContext ? IMAGE_STEPS_EDIT : IMAGE_STEPS_GENERATE,
              ...(editContext ? { inputImageUrl: editContext.source, imageEditStrength: editContext.strength } : {}),
              ...(!editContext && usedSettings.imageVariation ? { imageVariation: usedSettings.imageVariation } : {}),
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
        generateSubmissionRef.current = null
        reset()
      } catch (err) {
        logger.error(`Failed to persist generated image asset(s): ${err}`)
      } finally {
        addingImagesRef.current = false
      }
    })()
  }, [imagePaths, generatedSeed, currentProjectId, isGenerating, addAsset, lastPrompt, settings, reset])
  
  // Single writer for the recovery marker so the per-mode branches below can't drift. Captures
  // whatever generation id is active RIGHT NOW, before this generation starts, as baselineId —
  // see GenerationRecoveryContext for why: the handler that starts a generation loads its
  // pipeline (can take many seconds) before it ever reports a new id, so without a baseline
  // captured up front, a poll can't tell "my generation hasn't started reporting yet" apart from
  // "a stale, unrelated result predates this marker entirely".
  const writeRecoveryContext = async (
    ctx: Omit<GenerationRecoveryContext, 'projectId' | 'baselineId' | 'canCancel'>,
  ) => {
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
    const canCancel = ctx.genType === 'enhance'
      ? false
      : mode === 'ic-lora'
        ? true
        : canCancelLocalJob(
          ctx.genType === 'image' ? 'image' : 'video',
          shouldVideoGenerateWithLtxApi,
          shouldImageGenerateWithFalApi,
        )
    logger.info(`Writing recovery marker for ${currentProjectId} (genType=${ctx.genType ?? 'video'}, baselineId=${baselineId}, canCancel=${canCancel})`)
    localStorage.setItem(
      GENERATION_RECOVERY_KEY,
      JSON.stringify({
        projectId: currentProjectId,
        baselineId,
        ...ctx,
        canCancel,
      } satisfies GenerationRecoveryContext),
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
  const {
    isRunning: isOtherGenerationRunning,
  } = useGlobalGenerationLock()
  const isGenerationInProgressForEnhance = mode === 'ic-lora' ? isIcLoraGenerating : isGenerating
  const hasEnhanceText = prompt.trim().length > 0
  const hasEnhanceImage = mode === 'multi-keyframe'
    ? keyframes.length > 0
    : (mode === 'video' || mode === 'image') && !!inputImage
  const canEnhancePrompt = enhanceAvailableForMode
    && (enhanceProvider === 'api' ? hasGeminiApiKey : true)
    && (hasEnhanceText || hasEnhanceImage) && !isGenerationInProgressForEnhance && !isOtherGenerationRunning
  const enhanceBlockedByMissingGeminiKey = isEnhanceBlockedByMissingGeminiKey({
    enhanceAvailableForMode,
    enhanceProvider,
    hasGeminiApiKey,
    hasEnhanceInput: hasEnhanceText || hasEnhanceImage,
    isGenerationInProgressForEnhance,
    isOtherGenerationRunning,
  })
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
    // video (icLoraInput.videoPath), never this. Multi-keyframe uses the timeline stills
    // instead, so leftover first/last-frame chips must not leak into that enhance call.
    const enhanceKeyframes = mode === 'multi-keyframe'
      ? enhanceKeyframesPayload(keyframes)
      : undefined
    const imagePathForEnhance = mode === 'multi-keyframe'
      ? undefined
      : (mode === 'image' || mode === 'video' ? inputImage ?? undefined : undefined)
    const lastImagePathForEnhance = mode === 'multi-keyframe'
      ? undefined
      : (mode === 'video' && imagePathForEnhance
        ? inputLastImage ?? undefined
        : undefined)

    // Local-provider Enhance runs the same GIL-holding Gemma text encoder as local video/image
    // generation (see electron/python-backend.ts) — needs the same liveness-kill suppression.
    const result = await withGenerationActive(() => ApiClient.enhancePrompt({
      prompt: sourcePrompt,
      loraCatalogIds,
      icLoraId: mode === 'ic-lora' ? selectedIcLoraId ?? undefined : undefined,
      conditioningType,
      imagePath: imagePathForEnhance,
      lastImagePath: lastImagePathForEnhance,
      keyframes: enhanceKeyframes,
      ...(mode === 'multi-keyframe'
        ? {
            duration: settings.duration ?? undefined,
            fps: settings.fps,
          }
        : {}),
      provider: enhanceProvider,
      mediaType: mode === 'image' ? 'image' : 'video',
    }))

    setIsEnhancingPrompt(false)
    if (!result.ok) {
      if (result.error.code === 'GEMINI_INVALID_API_KEY' || result.error.code === 'GEMINI_API_KEY_MISSING') {
        window.dispatchEvent(new CustomEvent('open-settings', {
          detail: GEMINI_KEY_REQUIRED_SETTINGS_DETAIL,
        }))
      }
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
  }, [isEnhancingPrompt, mode, selectedLoras, selectedIcLoraId, icLoraCondType, inputImage, inputLastImage, keyframes, settings.duration, settings.fps, enhanceProvider, applyEnhanceResult, writeRecoveryContext])

  const handleEnhanceProviderChange = useCallback((provider: EnhanceProvider) => {
    setEnhanceProviderPref(provider)
    if (provider === 'api' && !hasGeminiApiKey) {
      window.dispatchEvent(new CustomEvent('open-settings', {
        detail: GEMINI_KEY_REQUIRED_SETTINGS_DETAIL,
      }))
    }
  }, [setEnhanceProviderPref, hasGeminiApiKey])

  const handleEnhancePrompt = useCallback(() => {
    if (enhanceBlockedByMissingGeminiKey) {
      window.dispatchEvent(new CustomEvent('open-settings', {
        detail: GEMINI_KEY_REQUIRED_SETTINGS_DETAIL,
      }))
      return
    }
    if (!canEnhancePrompt) return
    void runEnhance(prompt)
  }, [enhanceBlockedByMissingGeminiKey, canEnhancePrompt, prompt, runEnhance])

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
        model: retakeModel,
        input: {
          videoPath: retakeInput.videoPath,
          startTime: retakeInput.startTime,
          duration: retakeInput.duration,
          videoDuration: retakeInput.videoDuration,
        },
      }
      await writeRecoveryContext({ prompt, model: retakeModel })
      await submitRetake({
        videoPath: retakeInput.videoPath,
        startTime: retakeInput.startTime,
        duration: retakeInput.duration,
        prompt,
        mode: 'replace_audio_and_video',
        resolution: resolveResolution(retakeResolutionOpts, retakeResolutionKey),
        model: retakeModel,
      })
      return
    }

    if (mode === 'extend') {
      if (!extendInput.videoPath || !extendInput.ready) return
      extendSubmissionRef.current = {
        prompt,
        model: extendModel,
        input: {
          videoPath: extendInput.videoPath,
          direction: extendDirection,
          duration: extendSeconds,
          videoDuration: extendInput.videoDuration,
        },
      }
      await writeRecoveryContext({ prompt, model: extendModel })
      await submitExtend({
        videoPath: extendInput.videoPath,
        duration: extendSeconds,
        prompt,
        mode: extendDirection,
        resolution: resolveResolution(extendResolutionOpts, extendResolutionKey),
        model: extendModel,
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
      const imageSettings: GenerationSettings = {
        model: 'fast',
        duration: 5,
        videoResolution: settings.videoResolution,
        fps: 24,
        audio: false,
        cameraMotion: 'none',
        imageResolution: settings.imageResolution,
        imageAspectRatio: settings.aspectRatio ?? '16:9',
        // FORK: Krea 2 Turbo runs at IMAGE_STEPS_KREA2; Z-Image is fine at 4. Editing uses IMAGE_STEPS_EDIT.
        imageSteps: editSource ? IMAGE_STEPS_EDIT : (allowedImageModel(settings.imageModel) === 'krea-2-turbo' ? IMAGE_STEPS_KREA2 : IMAGE_STEPS_GENERATE),
        imageModel: allowedImageModel(settings.imageModel),
        variations: settings.variations,
        imageEditStrength: settings.imageEditStrength,
        imageVariation: settings.imageVariation,
      }
      const modelLabel = resolvePipelineDisplayName(videoModelSpecs, imageSettings.model) ?? undefined
      generateSubmissionRef.current = {
        kind: 'image',
        prompt,
        settings: imageSettings,
        modelLabel,
        inputImageUrl: editSource,
        inputLastImageUrl: null,
        inputAudioUrl: null,
      }
      await writeRecoveryContext({
        prompt,
        settings: imageSettings,
        modelLabel,
        genType: 'image',
        inputImageUrl: editSource ?? undefined,
      })
      generateImage(prompt, imageSettings, editSource)
    } else {
      // Generate video (t2v if no image/audio, i2v if image, a2v if audio)
      const imagePath = mode === 'multi-keyframe' ? null : inputImage || null
      const audioPath = genSpaceUsesAudioInput(mode) ? inputAudio || null : null
      const videoSettings = sanitizeVideoSettings(settings)
      if (!videoSettings) return
      const lastImagePath = imagePath && videoSettings.duration != null ? inputLastImage : null
      const genSettings: GenerationSettings = {
          ...videoSettings,
          model: videoSettings.model as VideoGenerationPipeline,
          imageModel: allowedImageModel(settings.imageModel),
          cameraMotion: 'none',
          imageAspectRatio: videoSettings.aspectRatio ?? '16:9',
          imageSteps: 4,
          // Local LoRA refs are filesystem paths the cloud API can't resolve.
          loras: mode !== 'multi-keyframe' && canUseUserLoras && selectedLoras.length > 0
            ? selectedLoras
            : undefined,
      }
      const modelLabel = resolvePipelineDisplayName(videoModelSpecs, genSettings.model) ?? undefined
      generateSubmissionRef.current = {
        kind: 'video',
        prompt,
        settings: genSettings,
        modelLabel,
        inputImageUrl: imagePath,
        inputLastImageUrl: lastImagePath,
        inputAudioUrl: audioPath,
        keyframes: mode === 'multi-keyframe' ? keyframes : undefined,
      }
      await writeRecoveryContext({
        prompt,
        settings: genSettings,
        modelLabel,
        inputImageUrl: imagePath ?? undefined,
        inputLastImageUrl: lastImagePath ?? undefined,
        inputAudioUrl: audioPath ?? undefined,
        keyframes: mode === 'multi-keyframe' ? toPersistedKeyframes(keyframes) : undefined,
      })
      generate(prompt, imagePath, genSettings, audioPath, lastImagePath, { mode, keyframes })
    }
  }
  
  // These asset-card handlers are memoized (stable identity) so the gallery grid
  // below can be memoized too — otherwise every prompt keystroke re-renders all
  // asset thumbnails, which grows into perceptible typing lag as clips pile up.
  const handleDelete = useCallback((assetId: string) => {
    if (currentProjectId) {
      deleteAsset(currentProjectId, assetId)
    }
  }, [currentProjectId, deleteAsset])

  const handleDragStart = useCallback((e: React.DragEvent, asset: Asset) => {
    e.dataTransfer.setData('asset', JSON.stringify(asset))
    e.dataTransfer.setData('assetId', asset.id)
    e.dataTransfer.effectAllowed = 'copy'
  }, [])

  const handleCreateVideo = useCallback((imageAsset: Asset) => {
    setMode('video')
    setInputImage(imageAsset.path)
    setPrompt(`${imageAsset.prompt || 'The scene comes to life...'}`)
  }, [])

  // Restore a past generation's full recipe into Gen Space so a re-run is one
  // click away — no re-picking the source image or retyping the prompt. The user
  // can tweak anything before hitting Generate; seed behaviour follows the
  // existing Lock Seed setting (unlocked = a fresh variation).
  const handleRegenerate = useCallback((asset: Asset) => {
    const params = asset.generationParams
    if (!params || !REGENERABLE_MODES.has(params.mode)) return
    const isImageGen = params.mode === 'text-to-image' || params.mode === 'image-edit'

    // Leave any retake/extend/IC-LoRA panel behind — this is a plain generation.
    setActiveRetakeSource(null)
    setActiveIcLoraSource(null)
    setMode(isImageGen ? 'image' : 'video')
    setPrompt(params.prompt)
    setInputImage(params.inputImageUrl ?? null)
    setInputAudio(params.inputAudioUrl ?? null)
    setSettings((prev) => sanitizeVideoSettings({
      ...prev,
      model: params.model || prev.model,
      duration: params.duration || prev.duration,
      videoResolution: params.resolution || prev.videoResolution,
      fps: params.fps || prev.fps,
      audio: params.audio,
      aspectRatio: params.imageAspectRatio || prev.aspectRatio,
      ...(isImageGen ? {
        imageVariation: params.imageVariation ?? 0,
        // Image assets store the placeholder model 'fast'; the real image model is in modelLabel.
        ...(params.modelLabel === 'Krea 2 Turbo' ? { imageModel: allowedImageModel('krea-2-turbo') }
          : params.modelLabel === 'Z-Image Turbo' ? { imageModel: 'z-image-turbo' } : {}),
      } : {}),
    }))
    // Reproduce exactly: lock the asset's seed (the prompt bar shows Locked; one click
    // back to Random). Older assets without a saved seed leave the seed setting alone.
    if (params.seed != null) updateAppSettings({ seedLocked: true, lockedSeed: params.seed })
    // Saved LoRA refs are models-dir-relative; re-select only those still installed.
    setSelectedLoras(
      (params.loras ?? []).flatMap((saved) => {
        const match = loraLibrary.items.find(
          (e) => e.installedPath
            && toModelsDirRelativeRef(e.installedPath, appSettings.modelsDir) === saved.ref,
        )
        return match?.installedPath
          ? [{ ref: match.installedPath, name: saved.name, scale: saved.scale }]
          : []
      }),
    )
  }, [sanitizeVideoSettings, loraLibrary.items, appSettings.modelsDir, updateAppSettings])

  const handleEditImage = (imageAsset: Asset) => {
    setMode('image')
    setInputImage(imageAsset.path)
    setPrompt((prev) => (prev.trim() ? prev : imageAsset.prompt || ''))
  }

  const handleRetake = useCallback((videoAsset: Asset) => {
    if (!canUseRetake) return
    setMode('retake')
    setPrompt('')
    setActiveRetakeSource(null)
    setRetakeResolutionKey('original')
    setRetakeModel(retakeExtendModelFromPipeline(videoAsset.generationParams?.model))
    setRetakeInitial({
      videoPath: videoAsset.path,
      duration: videoAsset.duration,
    })
    setRetakePanelKey((prev) => prev + 1)
  }, [])

  const handleExtend = useCallback((videoAsset: Asset) => {
    if (!canUseExtend) return
    setMode('extend')
    setPrompt('')
    setExtendDirection('end')
    setExtendSeconds(DEFAULT_EXTEND_SECONDS)
    setExtendResolutionKey('original')
    setExtendModel(retakeExtendModelFromPipeline(videoAsset.generationParams?.model))
    setExtendInitial({
      videoPath: videoAsset.path,
      duration: videoAsset.duration,
    })
    setExtendPanelKey((prev) => prev + 1)
  }, [canUseExtend])

  // "Continue as new shot": extract the clip's last frame, seed a fresh i2v gen
  // with it, and match the source's geometry/fps. The user tweaks the prompt and
  // hits Generate; the completion effect above trims + saves to Continuations.
  const handleContinue = useCallback(async (videoAsset: Asset, seekTime?: number) => {
    if (videoAsset.type !== 'video') return
    const api = window.electronAPI
    if (!api) return
    try {
      const res = await api.continuationExtractLastFrame({ videoPath: videoAsset.path, seekTime })
      setMode('video')
      setInputImage(res.framePath)
      continuationSeedRef.current = res.framePath
      setPrompt(videoAsset.prompt || '')
      setLastPrompt(videoAsset.prompt || '')
      setSettings((prev) => ({
        ...prev,
        aspectRatio: continuationAspectRatio(res.width, res.height),
        // Use the source clip's stored resolution tier, not one derived from its pixel
        // dimensions: a "540p" 16:9 gen outputs at 1024x576, and mapping short-side 576
        // by threshold lands it in 720p (the tier's actual short side varies by aspect,
        // e.g. 21:9 540p is 512, so a pixel-threshold map can't be exact). Fall back to
        // the pixel derivation only for a source with no stored tier (e.g. an import).
        videoResolution: videoAsset.resolution || continuationResolution(Math.min(res.width, res.height)),
        fps: Math.round(res.fps) || prev.fps,
      }))
      continuationPendingRef.current = true
    } catch (err) {
      logger.error(`Continue-from-last-frame setup failed: ${err}`)
    }
  }, [])

  const handleIcLora = useCallback((videoAsset: Asset) => {
    if (!canUseIcLora) return
    setMode('ic-lora')
    setPrompt('')
    setActiveIcLoraSource(null)
    setIcLoraInitial({ videoPath: videoAsset.path })
    setIcLoraPanelKey((prev) => prev + 1)
  }, [forceApiGenerations, canUseIcLora])

  const isRetakeMode = mode === 'retake'
  const isExtendMode = mode === 'extend'
  const isIcLoraMode = mode === 'ic-lora'
  const hasCompatibleVideoSettings = (mode !== 'video' && mode !== 'multi-keyframe') || (
    !isLoadingVideoGenerationModelSpecs
    && videoModelSpecs.length > 0
    && resolveVideoGenerationOptions({
      settings,
      modelSpecs: videoModelSpecs,
      hasAudio: genSpaceUsesAudioInput(mode) && Boolean(inputAudio),
      minimumDuration: shouldVideoGenerateWithLtxApi ? GENSPACE_MIN_SELECTABLE_DURATION_S : undefined,
    }).hasCompatibleOptions
  )
  // One global backend slot: Stop / Generate-disable must follow the in-flight job, not the
  // GenSpace mode tab. Retake/extend/IC-LoRA live in different hooks than video/image.
  // After a UI refresh those hooks remount at false; the same progress poll that disables
  // Generate (isOtherGenerationRunning) is the SSOT for "slot busy" / Stop.
  const slotBusyLocally = isGenerating || isRetaking || isExtending || isIcLoraGenerating
  const canSubmit = !isOtherGenerationRunning && !slotBusyLocally && (isRetakeMode
    ? retakeInput.ready && !!retakeInput.videoPath
    : isExtendMode
      ? extendInput.ready && !!extendInput.videoPath
      : isIcLoraMode
        ? (!!prompt.trim() || promptOptional) && icLoraInput.ready && !!icLoraInput.videoPath
          && (isCatalogIcLora || icLoraCondType !== 'custom' || !!icLoraCustomRef)
        : !!prompt.trim()
          && (mode !== 'multi-keyframe' || keyframes.length >= 1)
          && hasCompatibleVideoSettings)
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

  const bins = useMemo(() => activeProject?.bins ?? {}, [activeProject?.bins])
  const filteredAssets = useMemo(() => assets.filter(a => {
    if (showFavorites && !a.favorite) return false
    if (binFilter === ALL_BINS_FILTER) return true
    if (binFilter === null) return !a.binId
    return a.binId === binFilter
  }), [assets, showFavorites, binFilter])
  const favoriteCount = useMemo(() => assets.filter(a => a.favorite).length, [assets])
  const isLibraryMode = mode === 'video' || mode === 'image'

  // ── Multi-select helpers ──────────────────────────────────────────────
  const clearSelection = useCallback(() => setSelectedIds(new Set()), [])

  const toggleSelect = useCallback((id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const selectAllVisible = useCallback(() => {
    setSelectedIds(new Set(filteredAssets.map(a => a.id)))
  }, [filteredAssets])

  const exitSelectMode = useCallback(() => {
    setSelectMode(false)
    setSelectedIds(new Set())
    setMarquee(null)
    marqueeStartRef.current = null
  }, [])

  // Assign an existing tag (or clear) across every selected asset at once.
  const applyTagToSelection = useCallback((binId: string | null) => {
    if (!currentProjectId) return
    for (const id of selectedIds) {
      updateAsset(currentProjectId, id, { binId: binId ?? undefined })
    }
  }, [currentProjectId, selectedIds, updateAsset])

  const deleteSelection = useCallback(() => {
    if (selectedIds.size === 0) return
    if (!window.confirm(`Delete ${selectedIds.size} selected item${selectedIds.size === 1 ? '' : 's'}? This can't be undone.`)) return
    for (const id of selectedIds) handleDelete(id)
    setSelectedIds(new Set())
  }, [selectedIds, handleDelete])

  // Create a new tag from the modal, then route it to whatever asked for it:
  // the chip bar (create only), a single asset, or the whole multi-selection.
  const commitNewTag = useCallback(() => {
    const name = newTagName.trim()
    const target = creatingTagFor
    if (!name || !currentProjectId || target === null) return
    const binId = createBin(currentProjectId, name)
    if (target === '__selection__') {
      for (const id of selectedIds) updateAsset(currentProjectId, id, { binId })
    } else if (target !== '__bar__') {
      updateAsset(currentProjectId, target, { binId })
    }
    setCreatingTagFor(null)
    setNewTagName('')
  }, [newTagName, creatingTagFor, currentProjectId, createBin, selectedIds, updateAsset])

  // Marquee (rubber-band) drag-select over the grid. Coordinates are kept in the
  // scroll container's content space (viewport offset + scroll) so the overlay
  // and hit-testing stay correct while the grid scrolls.
  const MARQUEE_THRESHOLD = 5
  const computeMarqueeHits = useCallback((box: { left: number; top: number; right: number; bottom: number }) => {
    const container = gridScrollRef.current
    if (!container) return [] as string[]
    const crect = container.getBoundingClientRect()
    const hits: string[] = []
    container.querySelectorAll<HTMLElement>('[data-asset-id]').forEach(el => {
      const r = el.getBoundingClientRect()
      const left = r.left - crect.left + container.scrollLeft
      const top = r.top - crect.top + container.scrollTop
      const right = left + r.width
      const bottom = top + r.height
      const intersects = left < box.right && right > box.left && top < box.bottom && bottom > box.top
      if (intersects) {
        const id = el.getAttribute('data-asset-id')
        if (id) hits.push(id)
      }
    })
    return hits
  }, [])

  const handleMarqueePointerDown = useCallback((e: React.PointerEvent) => {
    if (!selectMode || e.button !== 0) return
    const container = gridScrollRef.current
    if (!container) return
    const crect = container.getBoundingClientRect()
    marqueeStartRef.current = {
      x: e.clientX - crect.left + container.scrollLeft,
      y: e.clientY - crect.top + container.scrollTop,
    }
    // Additive when holding shift/ctrl: keep the existing selection as the base.
    marqueeBaseRef.current = (e.shiftKey || e.ctrlKey || e.metaKey) ? new Set(selectedIds) : new Set()
    marqueeDraggedRef.current = false
  }, [selectMode, selectedIds])

  const handleMarqueePointerMove = useCallback((e: React.PointerEvent) => {
    const start = marqueeStartRef.current
    const container = gridScrollRef.current
    if (!start || !container) return
    const crect = container.getBoundingClientRect()
    const x = e.clientX - crect.left + container.scrollLeft
    const y = e.clientY - crect.top + container.scrollTop
    if (!marqueeDraggedRef.current) {
      if (Math.abs(x - start.x) < MARQUEE_THRESHOLD && Math.abs(y - start.y) < MARQUEE_THRESHOLD) return
      marqueeDraggedRef.current = true
      container.setPointerCapture?.(e.pointerId)
    }
    const box = {
      left: Math.min(start.x, x),
      top: Math.min(start.y, y),
      right: Math.max(start.x, x),
      bottom: Math.max(start.y, y),
    }
    setMarquee({ left: box.left, top: box.top, width: box.right - box.left, height: box.bottom - box.top })
    const hits = computeMarqueeHits(box)
    setSelectedIds(new Set([...marqueeBaseRef.current, ...hits]))
  }, [computeMarqueeHits])

  const handleMarqueePointerUp = useCallback(() => {
    if (marqueeDraggedRef.current) suppressCardClickRef.current = true
    marqueeStartRef.current = null
    marqueeDraggedRef.current = false
    setMarquee(null)
  }, [])

  // Memoized so typing in the prompt box (which re-renders GenSpace) doesn't
  // rebuild every thumbnail. Recomputes only when the assets or a card handler
  // actually change — all deps are stable (useCallback/useMemo/stable setters).
  const assetCards = useMemo(() => filteredAssets.map(asset => (
    <AssetCard
      key={asset.id}
      asset={asset}
      onDelete={() => handleDelete(asset.id)}
      onPlay={() => setSelectedAsset(asset)}
      onDragStart={handleDragStart}
      onCreateVideo={handleCreateVideo}
      onRegenerate={handleRegenerate}
      onEditImage={handleEditImage}
      onRetake={handleRetake}
      onExtend={handleExtend}
      onContinue={handleContinue}
      onIcLora={!forceApiGenerations ? handleIcLora : undefined}
      onToggleFavorite={() => currentProjectId && toggleFavorite(currentProjectId, asset.id)}
      bins={bins}
      onTag={(binId) => currentProjectId && updateAsset(currentProjectId, asset.id, { binId: binId ?? undefined })}
      onRequestNewTag={() => setCreatingTagFor(asset.id)}
      selectMode={selectMode}
      selected={selectedIds.has(asset.id)}
      onToggleSelect={() => {
        if (suppressCardClickRef.current) { suppressCardClickRef.current = false; return }
        toggleSelect(asset.id)
      }}
    />
  )), [
    filteredAssets, bins, handleDelete, handleDragStart, handleCreateVideo, handleRegenerate,
    handleRetake, handleExtend, handleIcLora, forceApiGenerations, currentProjectId,
    toggleFavorite, updateAsset, setSelectedAsset, setCreatingTagFor,
    selectMode, selectedIds, toggleSelect,
  ])

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

  useEffect(() => {
    if (!selectedAsset) return
    if (!filteredAssets.some(asset => asset.id === selectedAsset.id)) {
      setSelectedAsset(null)
    }
  }, [filteredAssets, selectedAsset])

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

      {/* Empty bin filter (untagged view or a specific folder with nothing in it) */}
      {isLibraryMode && !showFavorites && filteredAssets.length === 0 && assets.length > 0 && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none">
          <Tag className="h-12 w-12 text-zinc-700 mb-4" />
          <h3 className="text-lg font-semibold text-white mb-2">
            {binFilter === null ? 'Everything is tagged' : 'Nothing in this tag yet'}
          </h3>
          <p className="text-zinc-500 text-sm">
            {binFilter === null
              ? 'Click "All" above to see every asset.'
              : 'Use the tag icon on an asset to file it here.'}
          </p>
        </div>
      )}

      {/* Assets area — full width, no background, above the prompt bar */}
      {/* Kept mounted even with no assets so the Browse LoRAs / Favorites / size toolbar survives the empty state. */}
      {isLibraryMode && (
        <div className="absolute inset-x-0 top-0 bottom-[160px] flex flex-col px-4 pt-4">
          {/* Tag/folder chip bar */}
          <div className="flex items-center gap-1.5 pb-2 overflow-x-auto">
            <button
              onClick={() => setBinFilter(null)}
              className={`shrink-0 px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                binFilter === null
                  ? 'bg-white/20 text-white'
                  : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
              }`}
            >
              Untagged
            </button>
            <button
              onClick={() => setBinFilter(ALL_BINS_FILTER)}
              className={`shrink-0 px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                binFilter === ALL_BINS_FILTER
                  ? 'bg-white/20 text-white'
                  : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
              }`}
            >
              All
            </button>
            {Object.entries(bins).map(([binId, binName]) => (
              <div
                key={binId}
                className={`shrink-0 flex items-center gap-1 pl-3 pr-1.5 py-1 rounded-full text-xs font-medium transition-colors ${
                  binFilter === binId
                    ? 'bg-blue-500/30 text-blue-200 border border-blue-500/40'
                    : 'text-zinc-400 hover:text-white hover:bg-zinc-800 border border-transparent'
                }`}
              >
                <button onClick={() => setBinFilter(binId)}>{binName}</button>
                {binFilter === binId && (
                  <button
                    title="Delete this tag/folder"
                    onClick={() => {
                      if (!currentProjectId) return
                      if (!window.confirm(`Delete the "${binName}" tag? Assets in it will become untagged again.`)) return
                      deleteBin(currentProjectId, binId)
                      setBinFilter(null)
                    }}
                    className="rounded-full p-0.5 hover:bg-white/20"
                  >
                    <X className="h-3 w-3" />
                  </button>
                )}
              </div>
            ))}
            <button
              onClick={() => setCreatingTagFor('__bar__')}
              className="shrink-0 flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium text-zinc-500 hover:text-white hover:bg-zinc-800 transition-colors"
            >
              + New Tag
            </button>
          </div>

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
              onClick={() => { if (selectMode) exitSelectMode(); else setSelectMode(true) }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                selectMode
                  ? 'bg-accent/20 text-accent border border-accent/40'
                  : 'text-zinc-400 hover:text-white hover:bg-zinc-800'
              }`}
              title="Select multiple items to tag or delete them together"
            >
              <CheckSquare className="h-4 w-4" />
              {selectMode ? 'Done' : 'Select'}
            </button>

            {/* Multi-select actions — inline by the Select button so they're never missed */}
            {selectMode && (
              <>
                <span className="text-sm font-medium text-white tabular-nums px-1">
                  {selectedIds.size} selected
                </span>
                <button
                  onClick={selectAllVisible}
                  className="px-2.5 py-1.5 rounded-md text-xs font-medium text-zinc-300 hover:text-white hover:bg-zinc-800 transition-colors"
                >
                  Select all
                </button>
                <button
                  onClick={clearSelection}
                  disabled={selectedIds.size === 0}
                  className="px-2.5 py-1.5 rounded-md text-xs font-medium text-zinc-300 hover:text-white hover:bg-zinc-800 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Clear
                </button>
                <div className={selectedIds.size === 0 ? 'opacity-40 pointer-events-none' : ''}>
                  <SettingsDropdown
                    title="Tag selected"
                    value=""
                    placement="below"
                    onChange={(value) => {
                      if (value === '__new__') setCreatingTagFor('__selection__')
                      else if (value === '__none__') applyTagToSelection(null)
                      else applyTagToSelection(value)
                    }}
                    trigger={
                      <span className="flex items-center gap-1.5 rounded-md text-xs font-semibold text-white">
                        <Tag className="h-3.5 w-3.5" />
                        Tag
                        <ChevronUp className="h-3 w-3 rotate-180" />
                      </span>
                    }
                    triggerClassName="bg-accent hover:bg-accent-dark"
                    options={[
                      { value: '__none__', label: 'Remove from folder' },
                      ...Object.entries(bins).map(([binId, name]) => ({ value: binId, label: name })),
                      { value: '__new__', label: '+ New Tag…' },
                    ]}
                  />
                </div>
                <button
                  onClick={deleteSelection}
                  disabled={selectedIds.size === 0}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-semibold bg-red-600 hover:bg-red-500 text-white transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  Delete
                </button>
                <div className="w-px h-5 bg-zinc-700 mx-0.5" />
              </>
            )}
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
          <div
            ref={gridScrollRef}
            className={`relative overflow-y-auto overflow-x-hidden [scrollbar-gutter:stable] flex-1 ${selectMode ? 'select-none' : ''}`}
            onPointerDown={selectMode ? handleMarqueePointerDown : undefined}
            onPointerMove={selectMode ? handleMarqueePointerMove : undefined}
            onPointerUp={selectMode ? handleMarqueePointerUp : undefined}
          >
            {/* Marquee (rubber-band) rectangle while drag-selecting */}
            {marquee && (
              <div
                className="absolute z-40 rounded-sm border border-accent bg-accent/15 pointer-events-none"
                style={{ left: marquee.left, top: marquee.top, width: marquee.width, height: marquee.height }}
              />
            )}
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
                    <p className="mt-1 flex items-center gap-1 text-xs text-zinc-500 tabular-nums">
                      <Clock className="h-3 w-3" />
                      {formatClock(elapsedMs)}
                    </p>
                    {progress > 0 && (
                      <div className="w-32 h-1 bg-zinc-800 rounded-full mt-2 overflow-hidden">
                        <div className="h-full bg-violet-500 transition-all" style={{ width: `${progress}%` }} />
                      </div>
                    )}
                    <button
                      onClick={() => void cancel()}
                      className="mt-4 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium text-zinc-300 bg-zinc-700/60 hover:bg-red-600/80 hover:text-white transition-colors"
                      title="Stop this generation"
                    >
                      <Square className="h-3 w-3 fill-current" />
                      Stop
                    </button>
                  </div>
                </div>
              )}
              {assetCards}
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
            onCancel={cancel}
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

      {/* Floating prompt panel — wider, responsive, centered. z-20 keeps it above
          the grid cards' render-time chips (z-10), which otherwise bleed through
          the prompt box when it scrolls under them or the textarea is enlarged.
          Stays below z-50 dropdowns/modals so those still layer on top. */}
      <div className="absolute bottom-5 left-1/2 z-20 w-[min(700px,calc(100%-2rem))] -translate-x-1/2">

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
          onModeChange={handleGenSpaceModeChange}
          canUseIcLora={!forceApiGenerations}
          allowUltrawideVideo={!shouldVideoGenerateWithLtxApi}
          canUseMultiKeyframe={canUseMultiKeyframe}
          canUseRetake={canUseRetake}
          canUseExtend={canUseExtend}
          canUseUserLoras={canUseUserLoras}
          loraCatalogIdsByPath={loraDisplayNames}
          imageUsesFalApi={shouldImageGenerateWithFalApi}
          enhanceAvailableForMode={enhanceAvailableForMode}
          canEnhancePrompt={canEnhancePrompt}
          enhanceBlockedByMissingGeminiKey={enhanceBlockedByMissingGeminiKey}
          isEnhancingPrompt={isEnhancingPrompt}
          enhancePromptError={enhancePromptError}
          onEnhancePrompt={handleEnhancePrompt}
          enhanceProvider={canToggleEnhanceProvider ? enhanceProvider : undefined}
          onEnhanceProviderChange={handleEnhanceProviderChange}
          canUndoPrompt={canUndoPrompt}
          onUndoPrompt={handleUndoPrompt}
          canRedoPrompt={canRedoPrompt}
          onRedoPrompt={handleRedoPrompt}
          inputLastImage={inputLastImage}
          onInputLastImageChange={setInputLastImage}
          keyframes={keyframes}
          onKeyframesChange={setKeyframes}
          multiKeyframeMaxCount={multiKeyframeMaxCount}
          playheadFrame={playheadFrame}
          onPlayheadChange={setPlayheadFrame}
          onDragFrameChange={setDragFrame}
          prompt={prompt}
          onPromptChange={setPrompt}
          onClearPrompt={clearGenSpacePrompt}
          lastSeed={lastSeed}
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
          availableLoras={localLoras}
          selectedLoras={selectedLoras}
          onSelectedLorasChange={setSelectedLoras}
          loraDisplayNames={loraDisplayNames}
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
              <div className="flex items-center gap-1">
                <button
                  onClick={() => {
                    if (selectedAsset.type === 'video') void saveVideoFile(selectedAsset.path, selectedAsset.prompt)
                    else void saveImageFile(selectedAsset.path, selectedAsset.prompt)
                  }}
                  className="p-2 rounded-md text-zinc-400 hover:text-white transition-colors"
                  title={selectedAsset.type === 'video' ? 'Save video…' : 'Save image…'}
                >
                  <Download className="h-6 w-6" />
                </button>
                <button
                  onClick={() => setSelectedAsset(null)}
                  className="p-2 rounded-md text-zinc-400 hover:text-white transition-colors"
                  title="Close"
                >
                  <X className="h-6 w-6" />
                </button>
              </div>
            </div>

            {selectedAsset.type === 'video' ? (
              <video
                key={selectedAsset.id}
                ref={enlargedVideoRef}
                src={pathToFileUrl(selectedAsset.path)}
                controls
                autoPlay
                className="w-full rounded-xl object-contain max-h-[75vh]"
                onContextMenu={(e) => onVideoSaveContextMenu(e, {
                  sourcePath: selectedAsset.path,
                  name: selectedAsset.prompt,
                  onRegenerate: canRegenerateAsset(selectedAsset)
                    ? () => { setSelectedAsset(null); handleRegenerate(selectedAsset) }
                    : undefined,
                })}
              />
            ) : (
              <img
                key={selectedAsset.id}
                src={pathToFileUrl(selectedAsset.path)}
                alt=""
                className="w-full rounded-xl object-contain max-h-[75vh]"
                onContextMenu={(e) => onImageSaveContextMenu(e, {
                  sourcePath: selectedAsset.path,
                  name: selectedAsset.prompt,
                  onRegenerate: canRegenerateAsset(selectedAsset)
                    ? () => { setSelectedAsset(null); handleRegenerate(selectedAsset) }
                    : undefined,
                })}
              />
            )}
            {selectedAsset.type === 'video' && (
              <div className="mt-3 flex justify-center">
                <button
                  onClick={() => {
                    const asset = selectedAsset
                    if (!asset) return
                    // Grab the currently shown frame. The modal <video> autoplays,
                    // so it's usually AT the end when clicked — seeking exactly at
                    // duration decodes no frame (ffmpeg writes nothing), which is
                    // why this failed. Back off ~one frame from the end, matching
                    // saveVideoFrame's fix.
                    const v = enlargedVideoRef.current
                    let t: number | undefined
                    if (v) {
                      const dur = Number.isFinite(v.duration) && v.duration > 0 ? v.duration : 0
                      t = Number.isFinite(v.currentTime) ? Math.max(0, v.currentTime) : 0
                      if (dur > 0 && t > dur - 0.05) t = Math.max(0, dur - 0.05)
                    }
                    setSelectedAsset(null)
                    void handleContinue(asset, t)
                  }}
                  title="Seed a new clip from the exact frame shown here"
                  className="px-3 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium flex items-center gap-2 transition-colors"
                >
                  <Clapperboard className="h-4 w-4" />
                  Continue from this frame
                </button>
              </div>
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
                  selectedAsset.generationParams?.modelLabel,
                  selectedAsset.resolution,
                  selectedAsset.duration ? `${formatSeconds(selectedAsset.duration)}s` : 'Image',
                  selectedAsset.generationParams?.seed != null ? `Seed ${selectedAsset.generationParams.seed}` : null,
                  selectedAsset.generationParams?.imageVariation ? `Variation ${Math.round(selectedAsset.generationParams.imageVariation * 100)}%` : null,
                  selectedAsset.renderMs != null ? `${formatClock(selectedAsset.renderMs)} render` : null,
                ].filter(Boolean).join(' • ')}
              </p>
            </div>
          </div>
        </div>
      )}

      {videoSaveMenu}
      {imageSaveMenu}

      {creatingTagFor !== null && (
        <div
          className="fixed inset-0 z-[80] flex items-center justify-center bg-black/55"
          onClick={() => { setCreatingTagFor(null); setNewTagName('') }}
        >
          <div
            className="bg-zinc-900 border border-zinc-700 rounded-lg p-4 w-[280px]"
            onClick={(e) => e.stopPropagation()}
          >
            <p className="text-sm font-semibold text-white mb-2">New tag/folder</p>
            <input
              autoFocus
              value={newTagName}
              onChange={(e) => setNewTagName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  commitNewTag()
                } else if (e.key === 'Escape') {
                  setCreatingTagFor(null)
                  setNewTagName('')
                }
              }}
              placeholder="Tag name…"
              className="w-full rounded-md px-2 py-1.5 text-sm bg-zinc-800 text-white border border-zinc-700 outline-none focus:border-blue-500"
            />
            <div className="flex justify-end gap-2 mt-3">
              <button
                onClick={() => { setCreatingTagFor(null); setNewTagName('') }}
                className="rounded-md px-3 py-1.5 text-xs font-medium bg-zinc-800 text-white border border-zinc-700"
              >
                Cancel
              </button>
              <button
                onClick={commitNewTag}
                disabled={!newTagName.trim()}
                className="rounded-md px-3 py-1.5 text-xs font-medium bg-blue-600 text-white disabled:opacity-40"
              >
                Create
              </button>
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

