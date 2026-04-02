import { useState, useRef, useEffect, useCallback } from 'react'
import {
  Trash2, Download, Image, Video, X,
  Heart, Film, Volume2, VolumeX, Sparkles,
  Clock, Monitor, ChevronUp, Scissors, Music,
  ChevronLeft, ChevronRight, Copy, Check
} from 'lucide-react'
import { useProjects } from '../contexts/ProjectContext'
import type { GenSpaceRetakeSource } from '../contexts/ProjectContext'
import { useAppSettings } from '../contexts/AppSettingsContext'
import { useGeneration } from '../hooks/use-generation'
import { useRetake } from '../hooks/use-retake'
import { useIcLora } from '../hooks/use-ic-lora'
import type { ICLoraConditioningType } from '../components/ICLoraPanel'
import type { Asset } from '../types/project'
import { GenerationErrorDialog } from '../components/GenerationErrorDialog'
import { addVisualAssetToProject } from '../lib/asset-copy'
import { pathToFileUrl } from '../lib/file-url'
import {
  FORCED_API_VIDEO_FPS,
  FORCED_API_VIDEO_RESOLUTIONS,
  getAllowedForcedApiDurations,
  sanitizeForcedApiVideoSettings,
} from '../lib/api-video-options'
import { logger } from '../lib/logger'
import { RetakePanel } from '../components/RetakePanel'
import { ICLoraPanel, CONDITIONING_TYPES } from '../components/ICLoraPanel'
import { FreeApiKeyBubble } from '../components/FreeApiKeyBubble'

// Asset card with hover overlays
function AssetCard({
  asset,
  onDelete,
  onPlay,
  onDragStart,
  onCreateVideo,
  onRetake,
  onIcLora,
  onToggleFavorite
}: {
  asset: Asset
  onDelete: () => void
  onPlay: () => void
  onDragStart: (e: React.DragEvent, asset: Asset) => void
  onCreateVideo?: (asset: Asset) => void
  onRetake?: (asset: Asset) => void
  onIcLora?: (asset: Asset) => void
  onToggleFavorite?: () => void
}) {
  const hoverVideoRef = useRef<HTMLVideoElement>(null)
  const [isHovered, setIsHovered] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [isMuted, setIsMuted] = useState(true)
  const isFavorite = asset.favorite || false

  useEffect(() => {
    if (asset.type !== 'video') return
    if (!isHovered) {
      setCurrentTime(0)
      return
    }
    if (hoverVideoRef.current) {
      hoverVideoRef.current.muted = isMuted
      hoverVideoRef.current.play().catch(() => {})
    }
  }, [asset.type, isHovered, isMuted])

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
              <button
                onClick={(e) => { e.stopPropagation(); setIsMuted(!isMuted) }}
                className="p-1.5 rounded-lg bg-black/40 backdrop-blur-md text-white hover:bg-black/60 transition-colors"
              >
                {isMuted ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
              </button>
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

// Dropdown component for settings
function SettingsDropdown({ 
  trigger, 
  options, 
  value, 
  onChange,
  title 
}: { 
  trigger: React.ReactNode
  options: { value: string; label: string; disabled?: boolean; tooltip?: string; icon?: React.ReactNode }[]
  value: string
  onChange: (value: string) => void
  title: string
}) {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)
  
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isOpen])
  
  return (
    <div ref={dropdownRef} className="relative">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className={`flex shrink-0 items-center gap-1 whitespace-nowrap px-2 py-1.5 rounded-md transition-colors ${isOpen ? 'bg-zinc-700 hover:bg-zinc-700' : 'hover:bg-zinc-800'}`}
      >
        {trigger}
      </button>
      
      {isOpen && (
        <div className="absolute bottom-full left-0 mb-2 bg-zinc-800 border border-zinc-700 rounded-md p-2 min-w-[160px] shadow-xl z-[9999]">
          <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">{title}</div>
          <div className="space-y-1">
            {options.map(option => (
              <div key={option.value} className="relative group/option">
                <button
                  onClick={() => { if (!option.disabled) { onChange(option.value); setIsOpen(false) } }}
                  className={`w-full flex items-center justify-between px-2 py-2 rounded-md transition-colors text-left ${
                    option.disabled
                      ? 'cursor-not-allowed'
                      : value === option.value ? 'bg-white/20 hover:bg-white/25' : 'hover:bg-zinc-700'
                  }`}
                >
                  <span className={`flex items-center gap-2.5 text-sm ${
                    option.disabled 
                      ? 'text-zinc-600' 
                      : value === option.value ? 'text-white' : 'text-zinc-400'
                  }`}>
                    {option.icon && <span className="flex-shrink-0">{option.icon}</span>}
                    {option.label}
                  </span>
                  {value === option.value && !option.disabled && (
                    <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </button>
                {option.disabled && option.tooltip && (
                  <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 px-2 py-1 bg-zinc-700 rounded text-xs text-zinc-300 whitespace-nowrap opacity-0 group-hover/option:opacity-100 pointer-events-none z-[10000] transition-opacity">
                    {option.tooltip}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
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
  shouldVideoGenerateWithLtxApi,
  canGenerate,
  buttonLabel,
  buttonIcon,
  icLoraCondType,
  onIcLoraCondTypeChange,
  icLoraStrength,
  onIcLoraStrengthChange,
}: {
  mode: 'image' | 'video' | 'retake' | 'ic-lora'
  onModeChange: (mode: 'image' | 'video' | 'retake' | 'ic-lora') => void
  canUseIcLora: boolean
  prompt: string
  onPromptChange: (prompt: string) => void
  onGenerate: () => void
  isGenerating: boolean
  canGenerate: boolean
  buttonLabel: string
  buttonIcon: React.ReactNode
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
  }
  onSettingsChange: (settings: any) => void
  shouldVideoGenerateWithLtxApi: boolean
  icLoraCondType?: ICLoraConditioningType
  onIcLoraCondTypeChange?: (type: ICLoraConditioningType) => void
  icLoraStrength?: number
  onIcLoraStrengthChange?: (strength: number) => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const audioInputRef = useRef<HTMLInputElement>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [isAudioDragOver, setIsAudioDragOver] = useState(false)
  const isRetake = mode === 'retake'
  const isIcLora = mode === 'ic-lora'
  const LOCAL_MAX_DURATION: Record<string, number> = { '540p': 20, '720p': 10, '1080p': 5 }
  const localMaxDuration = LOCAL_MAX_DURATION[settings.videoResolution] ?? 20
  const videoDurationOptions = shouldVideoGenerateWithLtxApi
    ? [...getAllowedForcedApiDurations(settings.model, settings.videoResolution, settings.fps)]
    : [5, 6, 8, 10, 20].filter(d => d <= localMaxDuration)
  const videoResolutionOptions = shouldVideoGenerateWithLtxApi
    ? (inputAudio ? ['1080p'] : [...FORCED_API_VIDEO_RESOLUTIONS])
    : ['540p', '720p', '1080p']
  const videoFpsOptions = shouldVideoGenerateWithLtxApi ? [...FORCED_API_VIDEO_FPS] : [24, 25, 50]

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const assetData = e.dataTransfer.getData('asset')
    if (assetData) {
      const asset = JSON.parse(assetData) as Asset
      if (asset.type === 'image') {
        onInputImageChange(asset.path)
      }
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
    if (e.key === 'Enter' && !e.shiftKey && !isGenerating && canGenerate) {
      e.preventDefault()
      onGenerate()
    }
  }

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-2xl overflow-visible">
      {/* Top row: Image ref | Prompt | Generate */}
      <div className="flex items-start">
        {/* Input image drop zone — video mode only (I2V) */}
        {mode === 'video' && !isRetake && !isIcLora && (
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
              : mode === 'ic-lora'
                ? "Describe the style or transformation to apply..."
              : mode === 'image'
                ? "A close-up of a woman talking on the phone..."
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
          onChange={(v) => onModeChange(v as 'image' | 'video' | 'retake' | 'ic-lora')}
          options={[
            { value: 'image', label: 'Generate Images', icon: <Image className="h-4 w-4" /> },
            { value: 'video', label: 'Generate Videos', icon: <Video className="h-4 w-4" /> },
            { value: 'retake', label: 'Retake', icon: <Scissors className="h-4 w-4" /> },
            ...(canUseIcLora ? [{ value: 'ic-lora', label: 'IC-LoRA', icon: <Sparkles className="h-4 w-4" /> }] : []),
          ]}
          trigger={
            <>
              {mode === 'image' ? <Image className="h-3.5 w-3.5" /> : mode === 'retake' ? <Scissors className="h-3.5 w-3.5" /> : mode === 'ic-lora' ? <Sparkles className="h-3.5 w-3.5" /> : <Video className="h-3.5 w-3.5" />}
              <span className="text-zinc-300 font-medium">{mode === 'image' ? 'Image' : mode === 'retake' ? 'Retake' : mode === 'ic-lora' ? 'IC-LoRA' : 'Video'}</span>
              <ChevronUp className="h-3 w-3 text-zinc-500" />
            </>
          }
        />
        
        <div className="flex-1" />
        
        {isRetake ? (
          <div className="text-[10px] text-zinc-500 pr-2">Trim in the panel above, then retake</div>
        ) : isIcLora ? (
          <>
            <SettingsDropdown
              title="CONDITIONING TYPE"
              value={icLoraCondType || 'canny'}
              onChange={(v) => onIcLoraCondTypeChange?.(v as ICLoraConditioningType)}
              options={CONDITIONING_TYPES.map(ct => ({ value: ct.value, label: ct.label }))}
              trigger={
                <>
                  <span className="text-zinc-300 font-medium">{CONDITIONING_TYPES.find(ct => ct.value === icLoraCondType)?.label || 'Canny Edges'}</span>
                  <ChevronUp className="h-3 w-3 text-zinc-500" />
                </>
              }
            />
            <div className="w-px h-4 bg-zinc-700 mx-0.5" />
            <SettingsDropdown
              title="STRENGTH"
              value={String(icLoraStrength ?? 1.0)}
              onChange={(v) => onIcLoraStrengthChange?.(parseFloat(v))}
              options={[
                { value: '0.5', label: '0.50' },
                { value: '0.75', label: '0.75' },
                { value: '1', label: '1.00' },
                { value: '1.25', label: '1.25' },
                { value: '1.5', label: '1.50' },
                { value: '2', label: '2.00' },
              ]}
              trigger={
                <>
                  <span className="text-zinc-500 text-[10px]">STR</span>
                  <span className="text-zinc-300 font-medium">{(icLoraStrength ?? 1.0).toFixed(2)}</span>
                  <ChevronUp className="h-3 w-3 text-zinc-500" />
                </>
              }
            />
          </>
        ) : mode === 'image' ? (
          <>
            {/* Model indicator */}
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-zinc-800/50">
              <ZitIcon className="h-3.5 w-3.5" />
              <span className="text-zinc-300 font-medium">Z-Image Turbo</span>
            </div>
            
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
        ) : (
          <>
            <SettingsDropdown
              title="MODEL"
              value={settings.model}
              onChange={(v) => onSettingsChange({ ...settings, model: v })}
              options={
                shouldVideoGenerateWithLtxApi
                  ? [
                      { value: 'fast', label: 'LTX-2.3 Fast (API)', disabled: !!inputAudio, tooltip: inputAudio ? 'Fast model is not available for Audio-to-Video' : undefined },
                      { value: 'pro', label: 'LTX-2.3 Pro (API)' },
                    ]
                  : [
                      { value: 'fast', label: 'LTX 2.3 Fast' },
                    ]
              }
              trigger={
                <>
                  <LightricksIcon className="h-3.5 w-3.5" />
                  <span className="text-zinc-300 font-medium">
                    {shouldVideoGenerateWithLtxApi
                      ? (settings.model === 'pro' ? 'LTX-2.3 Pro (API)' : 'LTX-2.3 Fast (API)')
                      : 'LTX 2.3 Fast'}
                  </span>
                </>
              }
            />

            <div className="w-px h-4 bg-zinc-700 mx-0.5" />
            
            {/* Duration dropdown */}
            <SettingsDropdown
              title="DURATION"
              value={String(settings.duration)}
              onChange={(v) => onSettingsChange({ ...settings, duration: parseFloat(v) })}
              options={videoDurationOptions.map((value) => ({ value: String(value), label: `${value} Sec` }))}
              trigger={
                <>
                  <Clock className="h-3.5 w-3.5" />
                  <span>{settings.duration}s</span>
                </>
              }
            />
            
            {/* Resolution dropdown */}
            <SettingsDropdown
              title="RESOLUTION"
              value={settings.videoResolution}
              onChange={(v) => {
                const maxDur = LOCAL_MAX_DURATION[v] ?? 20
                const clampedDuration = settings.duration > maxDur ? maxDur : settings.duration
                onSettingsChange({ ...settings, videoResolution: v, duration: clampedDuration })
              }}
              options={videoResolutionOptions.map((value) => ({ value, label: value }))}
              trigger={
                <>
                  <Monitor className="h-3.5 w-3.5" />
                  <span>{settings.videoResolution.replace('p', '')}</span>
                </>
              }
            />

            {shouldVideoGenerateWithLtxApi && (
              <SettingsDropdown
                title="FPS"
                value={String(settings.fps)}
                onChange={(v) => onSettingsChange({ ...settings, fps: parseInt(v) })}
                options={videoFpsOptions.map((value) => ({ value: String(value), label: `${value}` }))}
                trigger={
                  <>
                    <Film className="h-3.5 w-3.5" />
                    <span>{settings.fps} FPS</span>
                  </>
                }
              />
            )}
            
            {/* Aspect Ratio dropdown */}
            <SettingsDropdown
              title="ASPECT RATIO"
              value={settings.aspectRatio}
              onChange={(v) => onSettingsChange({ ...settings, aspectRatio: v })}
              options={inputAudio
                ? [{ value: '16:9', label: '16:9' }]
                : [
                    { value: '16:9', label: '16:9' },
                    { value: '9:16', label: '9:16' },
                  ]
              }
              trigger={
                <>
                  <AspectIcon className="h-3.5 w-3.5" />
                  <span>{settings.aspectRatio}</span>
                </>
              }
            />
            
          </>
        )}
        
        {/* Generate button */}
        <button
          onClick={onGenerate}
          disabled={isGenerating || !canGenerate}
          className={`flex items-center gap-1.5 ml-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all flex-shrink-0 ${
            isGenerating || !canGenerate
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
}

export function GenSpace() {
  const {
    currentProject,
    currentProjectId,
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
  const { shouldVideoGenerateWithLtxApi, forceApiGenerations, settings: appSettings } = useAppSettings()
  const [mode, setMode] = useState<'image' | 'video' | 'retake' | 'ic-lora'>('video')
  const [prompt, setPrompt] = useState('')
  const [inputImage, setInputImage] = useState<string | null>(null)
  const [inputAudio, setInputAudio] = useState<string | null>(null)
  const [localError, setLocalError] = useState<string | null>(null)
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
    }
  } | null>(null)
  const [settings, setSettings] = useState(() => ({ ...DEFAULT_VIDEO_SETTINGS }))
  const applyForcedVideoSettings = useCallback(
    (next: { model: string; duration: number; videoResolution: string; fps: number; audio: boolean; aspectRatio: string; imageResolution: string; variations: number }) => {
      if (!shouldVideoGenerateWithLtxApi || mode !== 'video') return next
      return sanitizeForcedApiVideoSettings(next, { hasAudio: !!inputAudio })
    },
    [inputAudio, mode, shouldVideoGenerateWithLtxApi],
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
  } = useGeneration()

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
    ready: false,
  })
  const [retakePanelKey, setRetakePanelKey] = useState(0)
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
  })
  const [icLoraPanelKey, setIcLoraPanelKey] = useState(0)
  const [icLoraCondType, setIcLoraCondType] = useState<ICLoraConditioningType>('canny')
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
    setIcLoraPanelKey((prev) => prev + 1)
    setGenSpaceIcLoraSource(null)
  }, [genSpaceIcLoraSource, forceApiGenerations, setGenSpaceIcLoraSource])

  useEffect(() => {
    if (forceApiGenerations && mode === 'ic-lora') {
      setMode('video')
    }
  }, [forceApiGenerations, mode])

  useEffect(() => {
    if (!shouldVideoGenerateWithLtxApi || mode !== 'video') return
    setSettings((prev) => applyForcedVideoSettings({ ...prev, model: 'fast' }))
  }, [applyForcedVideoSettings, mode, shouldVideoGenerateWithLtxApi])

  useEffect(() => {
    if (retakeError) {
      setLocalError(retakeError)
    }
  }, [retakeError])

  useEffect(() => {
    if (icLoraError) {
      setLocalError(icLoraError)
    }
  }, [icLoraError])

  // Force pro model + resolution when audio is attached (A2V only supports pro @ 1080p 16:9)
  useEffect(() => {
    if (inputAudio) {
      setSettings(prev => applyForcedVideoSettings({ ...prev, model: 'pro', aspectRatio: '16:9' }))
    }
  }, [inputAudio]) // eslint-disable-line react-hooks/exhaustive-deps

  // Only show assets that were generated (have generationParams), not imported files
  const assets = (currentProject?.assets || []).filter(a => a.generationParams)
  const [lastPrompt, setLastPrompt] = useState('')
  
  // When video generation completes, add to project assets
  useEffect(() => {
    if (!videoPath || !currentProjectId || isGenerating) return

    const generationKey = videoPath
    if (persistedVideoKeyRef.current === generationKey) return
    persistedVideoKeyRef.current = generationKey

    const genMode = inputAudio
      ? 'audio-to-video'
      : inputImage ? 'image-to-video' : 'text-to-video'
    const savedVideoSettings = applyForcedVideoSettings(settings)

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
  }, [videoPath, currentProjectId, isGenerating, applyForcedVideoSettings, settings, inputImage, inputAudio, lastPrompt, addAsset, reset])

  // When retake completes, add as take or new asset
  useEffect(() => {
    if (!retakeResult || !currentProjectId || isRetaking) return
    const submission = retakeSubmissionRef.current
    if (!submission) return
    retakeSubmissionRef.current = null

    ;(async () => {
      const usedPrompt = submission.prompt
      const usedInput = submission.input
      const copied = await addVisualAssetToProject(retakeResult.videoPath, currentProjectId, 'video')
      if (!copied) {
        logger.error('Could not persist retake result to project storage')
        setLocalError('Failed to save retake output to project storage.')
        setActiveRetakeSource(null)
        resetRetake()
        return
      }

      if (activeRetakeSource?.assetId) {
        const sourceAsset = currentProject?.assets?.find(a => a.id === activeRetakeSource.assetId)
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
  }, [retakeResult, isRetaking, currentProjectId, currentProject?.assets, activeRetakeSource, addAsset, addTakeToAsset, setPendingRetakeUpdate, resetRetake])

  useEffect(() => {
    if (!icLoraResult || !currentProjectId || isIcLoraGenerating) return
    const submission = icLoraSubmissionRef.current
    if (!submission) return
    icLoraSubmissionRef.current = null

    ;(async () => {
      const copied = await addVisualAssetToProject(icLoraResult.videoPath, currentProjectId, 'video')
      if (!copied) {
        logger.error('Could not persist IC-LoRA result to project storage')
        setLocalError('Failed to save IC-LoRA output to project storage.')
        setActiveIcLoraSource(null)
        resetIcLora()
        return
      }

      if (activeIcLoraSource?.assetId) {
        const sourceAsset = currentProject?.assets?.find(a => a.id === activeIcLoraSource.assetId)
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
  }, [icLoraResult, isIcLoraGenerating, currentProjectId, currentProject?.assets, activeIcLoraSource, addAsset, addTakeToAsset, setPendingIcLoraUpdate])
  
  // When image generation/editing completes, add all images to project assets
  useEffect(() => {
    if (imagePaths.length > 0 && currentProjectId && !isGenerating) {
      const genMode = 'text-to-image'
      ;(async () => {
        for (let i = 0; i < imagePaths.length; i++) {
          const imgPath = imagePaths[i]
          const exists = assets.some(a => a.path === imgPath)
          if (!exists) {
            const copied = await addVisualAssetToProject(imgPath, currentProjectId, 'image')
            if (!copied) {
              logger.error(`Could not persist generated image to project storage: ${imgPath}`)
              continue
            }
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
                imageSteps: 4,
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
        }
      })()
    }
  }, [imagePaths, currentProjectId, isGenerating])
  
  const handleGenerate = async () => {
    if (mode === 'ic-lora') {
      if (!prompt.trim() || !icLoraInput.videoPath || !icLoraInput.ready) return
      icLoraSubmissionRef.current = {
        prompt,
        input: {
          videoPath: icLoraInput.videoPath,
          conditioningType: icLoraCondType,
          conditioningStrength: icLoraStrength,
        },
      }
      await submitIcLora({
        videoPath: icLoraInput.videoPath,
        conditioningType: icLoraCondType,
        conditioningStrength: icLoraStrength,
        prompt,
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
      await submitRetake({
        videoPath: retakeInput.videoPath,
        startTime: retakeInput.startTime,
        duration: retakeInput.duration,
        prompt,
        mode: 'replace_audio_and_video',
      })
      return
    }

    if (!prompt.trim()) return

    // Save the prompt before generation starts
    setLastPrompt(prompt)

    if (mode === 'image') {
      generateImage(
        prompt,
        {
          model: 'fast' as 'fast' | 'pro',
          duration: 5,
          videoResolution: settings.videoResolution,
          fps: 24,
          audio: false,
          cameraMotion: 'none',
          imageResolution: settings.imageResolution,
          imageAspectRatio: settings.aspectRatio,
          imageSteps: 4,
          variations: settings.variations,
        }
      )
    } else {
      // Generate video (t2v if no image/audio, i2v if image, a2v if audio)
      const imagePath = inputImage || null
      const audioPath = inputAudio || null
      const videoSettings = applyForcedVideoSettings(settings)
      if (audioPath) videoSettings.model = 'pro'

      generate(
        prompt,
        imagePath,
        {
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
        },
        audioPath,
      )
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

  const handleRetake = (videoAsset: Asset) => {
    setMode('retake')
    setPrompt('')
    setActiveRetakeSource(null)
    setRetakeInitial({
      videoPath: videoAsset.path,
      duration: videoAsset.duration,
    })
    setRetakePanelKey((prev) => prev + 1)
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
  const isIcLoraMode = mode === 'ic-lora'
  const canSubmit = isRetakeMode
    ? retakeInput.ready && !!retakeInput.videoPath && !isRetaking
    : isIcLoraMode
      ? !!prompt.trim() && icLoraInput.ready && !!icLoraInput.videoPath && !isIcLoraGenerating
      : !!prompt.trim()
  const promptButtonLabel = isRetakeMode ? 'Retake' : isIcLoraMode ? 'Generate' : 'Generate'
  const promptButtonIcon = isRetakeMode
    ? <Scissors className="h-3.5 w-3.5" />
    : isIcLoraMode
      ? <Sparkles className="h-3.5 w-3.5" />
    : <Sparkles className={`h-3.5 w-3.5 ${isGenerating ? 'animate-pulse' : ''}`} />
  const promptGenerating = isRetakeMode ? isRetaking : isIcLoraMode ? isIcLoraGenerating : isGenerating
  
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
      {isLibraryMode && (assets.length > 0 || isGenerating) && (
        <div className="absolute inset-x-0 top-0 bottom-[160px] flex flex-col px-4 pt-4">
          {/* Top bar */}
          <div className="flex items-center justify-end pb-2 gap-2">
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
                  onRetake={handleRetake}
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
            onChange={(data) => setRetakeInput(data)}
          />
        </div>
      )}

      {mode === 'ic-lora' && !forceApiGenerations && (
        <div className="absolute inset-x-0 top-0 bottom-[160px] px-4 pt-4 pb-4 flex flex-col overflow-hidden">
          <ICLoraPanel
            initialVideoPath={icLoraInitial.videoPath}
            resetKey={icLoraPanelKey}
            fillHeight
            isProcessing={isIcLoraGenerating}
            processingStatus={icLoraStatus}
            conditioningType={icLoraCondType}
            onConditioningTypeChange={setIcLoraCondType}
            conditioningStrength={icLoraStrength}
            onConditioningStrengthChange={setIcLoraStrength}
            outputVideoPath={icLoraResult?.videoPath || null}
            onChange={setIcLoraInput}
          />
        </div>
      )}

      {/* Floating prompt panel — wider, responsive, centered */}
      <div className="absolute bottom-5 left-1/2 w-[min(700px,calc(100%-2rem))] -translate-x-1/2">

        <FreeApiKeyBubble
          forceApiGenerations={forceApiGenerations}
          hasLtxApiKey={appSettings.hasLtxApiKey}
          isGenerating={isGenerating}
        />

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
          inputImage={inputImage}
          onInputImageChange={setInputImage}
          inputAudio={inputAudio}
          onInputAudioChange={setInputAudio}
          settings={settings}
          onSettingsChange={(nextSettings) => setSettings(applyForcedVideoSettings(nextSettings))}
          shouldVideoGenerateWithLtxApi={shouldVideoGenerateWithLtxApi}
          icLoraCondType={icLoraCondType}
          onIcLoraCondTypeChange={setIcLoraCondType}
          icLoraStrength={icLoraStrength}
          onIcLoraStrengthChange={setIcLoraStrength}
        />
      </div>
      
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
                <p className="text-zinc-300">{selectedAsset.prompt}</p>
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
                {selectedAsset.resolution} • {selectedAsset.duration ? `${selectedAsset.duration}s` : 'Image'}
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
