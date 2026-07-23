import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Film, Play, Pause, Volume2, VolumeX, Loader2, Upload, Trash2, RefreshCw } from 'lucide-react'
import { logger } from '../lib/logger'
import { pathToFileUrl } from '../lib/file-url'

// Shared video source + preview surface used by RetakePanel and ExtendPanel. Owns the
// video file (drop/browse/clear), playback, mute, the timecode transport bar, and the
// filmstrip thumbnail strip with click-to-seek. Mode-specific UI (retake's trim handles,
// extend's direction/seconds controls) is injected via the render-prop slots.

export interface VideoPreviewContext {
  videoPath: string | null
  videoUrl: string | null
  videoDuration: number
  currentTime: number
  isPlaying: boolean
  thumbnails: string[]
  videoRef: React.RefObject<HTMLVideoElement>
  filmstripRef: React.RefObject<HTMLDivElement>
  seek: (seconds: number) => void
}

interface VideoPreviewPanelProps {
  title: string
  initialVideoPath?: string | null
  initialDuration?: number
  resetKey?: number
  isProcessing?: boolean
  processingStatus?: string
  processingDefault?: string
  fillHeight?: boolean
  emptyTitle?: string
  hint?: { title: string; subtitle?: string }
  // Validation error for the current source; when set, shown as a banner.
  errorMessage?: string
  // Owned by a parent when it needs them in its own effects (e.g. retake drag/keyboard).
  videoRef?: React.RefObject<HTMLVideoElement>
  filmstripRef?: React.RefObject<HTMLDivElement>
  onSourceChange?: (data: { videoPath: string | null; videoDuration: number; width: number; height: number }) => void
  // Returns true when the parent is mid-interaction (e.g. dragging a trim handle) and the
  // filmstrip click-to-seek should be ignored — a drag release lands a click on the strip.
  shouldSuppressSeek?: () => boolean
  // Absolutely-positioned overlay rendered on top of the filmstrip thumbnails.
  filmstripOverlay?: (ctx: VideoPreviewContext) => React.ReactNode
  // Content rendered below the filmstrip (labels for retake, controls for extend).
  belowFilmstrip?: (ctx: VideoPreviewContext) => React.ReactNode
}

export function formatTimecode(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${String(m).padStart(2, '0')}:${s.toFixed(2).padStart(5, '0')}`
}

export function VideoPreviewPanel({
  title,
  initialVideoPath,
  initialDuration,
  resetKey,
  isProcessing = false,
  processingStatus = '',
  processingDefault = 'Processing...',
  fillHeight = false,
  emptyTitle = 'Drop a video',
  hint,
  errorMessage,
  videoRef: videoRefProp,
  filmstripRef: filmstripRefProp,
  onSourceChange,
  shouldSuppressSeek,
  filmstripOverlay,
  belowFilmstrip,
}: VideoPreviewPanelProps) {
  const internalVideoRef = useRef<HTMLVideoElement>(null)
  const internalFilmstripRef = useRef<HTMLDivElement>(null)
  const videoRef = videoRefProp ?? internalVideoRef
  const filmstripRef = filmstripRefProp ?? internalFilmstripRef

  const [videoPath, setVideoPath] = useState<string | null>(initialVideoPath || null)
  const videoUrl = videoPath ? pathToFileUrl(videoPath) : null
  const [videoDuration, setVideoDuration] = useState<number>(initialDuration || 0)

  const [isPlaying, setIsPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [isDragOver, setIsDragOver] = useState(false)
  const [dimensions, setDimensions] = useState<{ width: number; height: number }>({ width: 0, height: 0 })

  const [thumbnails, setThumbnails] = useState<string[]>([])
  const [thumbCount] = useState(20)
  const extractingRef = useRef(false)

  useEffect(() => {
    if (resetKey === undefined) return
    setVideoPath(initialVideoPath || null)
    setVideoDuration(initialDuration || 0)
    setIsPlaying(false)
    setCurrentTime(0)
    setThumbnails([])
    setDimensions({ width: 0, height: 0 })
    extractingRef.current = false
  }, [resetKey, initialVideoPath, initialDuration])

  useEffect(() => {
    if (!videoUrl) {
      setVideoDuration(0)
      setIsPlaying(false)
      setCurrentTime(0)
      setThumbnails([])
      setDimensions({ width: 0, height: 0 })
      extractingRef.current = false
    }
  }, [videoUrl, initialDuration])

  useEffect(() => {
    onSourceChange?.({ videoPath, videoDuration, width: dimensions.width, height: dimensions.height })
  }, [videoPath, videoDuration, dimensions, onSourceChange])

  useEffect(() => {
    if (!videoUrl || extractingRef.current || videoDuration <= 0) return
    extractingRef.current = true
    let cancelled = false

    const extractThumbnails = async () => {
      const video = document.createElement('video')
      video.crossOrigin = 'anonymous'
      video.preload = 'auto'
      video.muted = true
      video.src = videoUrl

      await new Promise<void>((resolve, reject) => {
        video.onloadeddata = () => resolve()
        video.onerror = () => reject(new Error('Failed to load video for thumbnails'))
        setTimeout(() => reject(new Error('Timeout loading video')), 10000)
      })
      if (cancelled) return

      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')!
      const thumbWidth = 80
      const thumbHeight = Math.round(thumbWidth * (video.videoHeight / video.videoWidth))
      canvas.width = thumbWidth
      canvas.height = thumbHeight

      const frames: string[] = []
      const count = Math.min(thumbCount, Math.max(5, Math.floor(videoDuration / 0.25)))

      for (let i = 0; i < count; i++) {
        const seekTime = (i / count) * videoDuration
        video.currentTime = seekTime
        await new Promise<void>(resolve => {
          video.onseeked = () => resolve()
          setTimeout(resolve, 500)
        })
        if (cancelled) { video.src = ''; video.load(); return }
        ctx.drawImage(video, 0, 0, thumbWidth, thumbHeight)
        frames.push(canvas.toDataURL('image/jpeg', 0.6))
      }

      video.src = ''
      video.load()
      setThumbnails(frames)
    }

    extractThumbnails().catch(err => {
      logger.warn(`Filmstrip extraction failed: ${err}`)
    })

    // Abort an in-flight extraction when the source changes or the component unmounts.
    // Reset the re-entry guard too: videoDuration is refined by loadedmetadata after the
    // asset's approximate duration kicks off the first run, and without this the cancelled
    // run leaves the guard stuck true and the restart never happens (permanent spinner).
    return () => {
      cancelled = true
      extractingRef.current = false
    }
  }, [videoUrl, videoDuration, thumbCount])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const handler = () => setCurrentTime(video.currentTime)
    const onLoaded = () => {
      // The real file duration is authoritative; initialDuration is only a pre-load hint
      // (an asset's stored duration can be stale/approximate and mismatch the actual file).
      if (video.duration && Number.isFinite(video.duration)) {
        setVideoDuration(video.duration)
      }
      if (video.videoWidth && video.videoHeight) {
        setDimensions({ width: video.videoWidth, height: video.videoHeight })
      }
    }
    video.addEventListener('timeupdate', handler)
    video.addEventListener('loadedmetadata', onLoaded)
    return () => {
      video.removeEventListener('timeupdate', handler)
      video.removeEventListener('loadedmetadata', onLoaded)
    }
  }, [videoUrl])

  const togglePlay = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) {
      video.play()
      setIsPlaying(true)
    } else {
      video.pause()
      setIsPlaying(false)
    }
  }, [])

  const toggleMute = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    video.muted = !video.muted
    setIsMuted(video.muted)
  }, [])

  // Shared transport keyboard: play/pause + frame/scrub. (Retake adds its own i/o keys.)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      const key = e.key.toLowerCase()
      const video = videoRef.current
      if (key === ' ') {
        e.preventDefault()
        e.stopPropagation()
        togglePlay()
      } else if (key === 'arrowleft') {
        e.preventDefault()
        e.stopPropagation()
        if (video) {
          video.pause()
          setIsPlaying(false)
          video.currentTime = Math.max(0, video.currentTime - 1 / 24)
        }
      } else if (key === 'arrowright') {
        e.preventDefault()
        e.stopPropagation()
        if (video) {
          video.pause()
          setIsPlaying(false)
          video.currentTime = Math.min(videoDuration, video.currentTime + 1 / 24)
        }
      } else if (key === 'k') {
        e.preventDefault()
        e.stopPropagation()
        if (video) { video.pause(); setIsPlaying(false) }
      } else if (key === 'l') {
        e.preventDefault()
        e.stopPropagation()
        if (video) { video.play(); setIsPlaying(true) }
      }
    }
    window.addEventListener('keydown', handler, true)
    return () => window.removeEventListener('keydown', handler, true)
  }, [togglePlay, videoDuration])

  const seek = useCallback((seconds: number) => {
    const video = videoRef.current
    if (video) video.currentTime = seconds
  }, [])

  const handleFilmstripClick = useCallback((e: React.MouseEvent) => {
    if (shouldSuppressSeek?.()) return
    const strip = filmstripRef.current
    const video = videoRef.current
    if (!strip || !video) return
    const rect = strip.getBoundingClientRect()
    const fraction = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    video.currentTime = fraction * videoDuration
  }, [videoDuration, shouldSuppressSeek])

  // Swap to a new source. Reset duration/dimensions too (not just thumbnails) so the next
  // onSourceChange doesn't emit the new path with the previous file's metadata until
  // loadedmetadata fires.
  const selectSource = useCallback((path: string) => {
    setVideoPath(path)
    setVideoDuration(0)
    setCurrentTime(0)
    setDimensions({ width: 0, height: 0 })
    setThumbnails([])
    extractingRef.current = false
  }, [])

  const handleBrowse = useCallback(async () => {
    const paths = await window.electronAPI.showOpenFileDialog({
      title: 'Select Video',
      filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'avi', 'webm', 'mkv'] }],
    })
    if (paths && paths.length > 0) {
      selectSource(paths[0])
    }
  }, [selectSource])

  const handleClear = useCallback(() => {
    setVideoPath(null)
    setVideoDuration(0)
    setIsPlaying(false)
    setCurrentTime(0)
    setThumbnails([])
    extractingRef.current = false
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const assetData = e.dataTransfer.getData('asset')
    if (assetData) {
      try {
        const asset = JSON.parse(assetData) as { type?: string; path?: string }
        if (asset.type === 'video' && asset.path) {
          selectSource(asset.path)
          return
        }
      } catch {
        // fall through to file handling
      }
    }

    const file = e.dataTransfer.files?.[0]
    if (file) {
      const filePath = window.electronAPI?.getPathForFile(file)
      if (filePath) {
        selectSource(filePath)
      }
    }
  }, [selectSource])

  const ctx: VideoPreviewContext = {
    videoPath,
    videoUrl,
    videoDuration,
    currentTime,
    isPlaying,
    thumbnails,
    videoRef,
    filmstripRef,
    seek,
  }

  const playheadFrac = videoDuration > 0 ? currentTime / videoDuration : 0

  return (
    <div className={`bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden flex flex-col ${fillHeight ? 'h-full min-h-0' : ''}`}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Film className="h-4 w-4 text-blue-400" />
          <span className="text-sm font-semibold text-white">{title}</span>
          {videoPath && (
            <span className="text-xs text-zinc-500 truncate max-w-[240px]">
              {videoPath.split(/[/\\]/).pop()}
            </span>
          )}
        </div>
        {videoUrl && (
          <div className="flex items-center gap-2">
            <button
              onClick={handleClear}
              className="p-1.5 rounded-md hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors"
              title="Clear video"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={handleBrowse}
              className="p-1.5 rounded-md hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors"
              title="Replace video"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          </div>
        )}
      </div>

      {!videoUrl ? (
        <div
          className={`p-8 flex flex-col items-center justify-center gap-3 border-2 border-dashed rounded-xl m-4 transition-colors ${
            isDragOver ? 'border-blue-500 bg-blue-500/10' : 'border-zinc-700'
          }`}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
        >
          <div className="p-3 rounded-full bg-zinc-800">
            <Upload className="h-5 w-5 text-zinc-400" />
          </div>
          <div className="text-center">
            <p className="text-sm text-white">{emptyTitle}</p>
            <p className="text-xs text-zinc-500">mp4, mov, avi, webm, mkv</p>
          </div>
          <button
            onClick={handleBrowse}
            className="px-4 py-1.5 text-xs font-medium rounded-md bg-white text-black hover:bg-zinc-200 transition-colors"
          >
            Browse
          </button>
        </div>
      ) : (
        <div className="flex-1 min-h-0 flex flex-col">
          <div className="relative bg-black flex-1 min-h-0">
            <video
              ref={videoRef}
              src={videoUrl}
              className="w-full h-full object-contain"
              onClick={togglePlay}
              onEnded={() => setIsPlaying(false)}
            />
            <div className="absolute bottom-2 left-2 flex items-center gap-1.5">
              <button
                onClick={toggleMute}
                className="p-1.5 rounded bg-black/60 hover:bg-black/80 text-white/80 hover:text-white transition-colors"
              >
                {isMuted ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
              </button>
            </div>
          </div>

          <div className="flex-shrink-0">
            <div className="flex items-center justify-center gap-3 px-4 py-2 bg-zinc-900 border-b border-zinc-800">
              <button
                onClick={togglePlay}
                className="p-1 rounded hover:bg-zinc-800 text-white transition-colors"
              >
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </button>
              <span className="text-xs font-mono text-zinc-400">
                {formatTimecode(currentTime)} / {formatTimecode(videoDuration)}
              </span>
            </div>

            {hint && (
              <div className="px-4 pt-3 pb-1">
                <p className="text-xs font-semibold text-white">{hint.title}</p>
                {hint.subtitle && <p className="text-[10px] text-zinc-500 mt-0.5">{hint.subtitle}</p>}
              </div>
            )}

            <div className="px-4 pb-4">
              <div className="relative h-3 mb-0">
                <div
                  className="absolute pointer-events-none z-10"
                  style={{ left: `${playheadFrac * 100}%`, transform: 'translateX(-50%)' }}
                >
                  <svg width="10" height="10" viewBox="0 0 10 10">
                    <polygon points="0,0 10,0 5,8" fill="#fff" />
                  </svg>
                </div>
              </div>
              <div
                ref={filmstripRef}
                className="relative h-14 rounded-md overflow-hidden cursor-pointer select-none"
                onClick={handleFilmstripClick}
              >
                <div className="absolute inset-0 flex">
                  {thumbnails.length > 0 ? (
                    thumbnails.map((thumb, i) => (
                      <img
                        key={i}
                        src={thumb}
                        alt=""
                        className="h-full flex-1 object-cover"
                        style={{ minWidth: 0 }}
                        draggable={false}
                      />
                    ))
                  ) : (
                    <div className="w-full h-full bg-zinc-800 flex items-center justify-center">
                      <Loader2 className="h-4 w-4 text-zinc-600 animate-spin" />
                    </div>
                  )}
                </div>

                {filmstripOverlay?.(ctx)}

                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-zinc-800 pointer-events-none z-[15]"
                  style={{ left: `${playheadFrac * 100}%` }}
                />
              </div>

              {belowFilmstrip?.(ctx)}
            </div>

            {errorMessage ? (
              <div className="px-4 pb-4">
                <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-red-600/10 border border-red-500/20">
                  <span className="text-xs text-red-300">{errorMessage}</span>
                </div>
              </div>
            ) : isProcessing ? (
              <div className="px-4 pb-4">
                <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-blue-600/10 border border-blue-500/20">
                  <Loader2 className="h-3.5 w-3.5 text-blue-400 animate-spin flex-shrink-0" />
                  <span className="text-xs text-blue-300">{processingStatus || processingDefault}</span>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  )
}
