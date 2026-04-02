import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Film, Play, Pause, Volume2, VolumeX, Loader2, Upload, Trash2, RefreshCw } from 'lucide-react'
import { logger } from '../lib/logger'
import { pathToFileUrl } from '../lib/file-url'

interface RetakePanelProps {
  initialVideoPath?: string | null
  initialDuration?: number
  resetKey?: number
  isProcessing?: boolean
  processingStatus?: string
  fillHeight?: boolean
  onChange?: (data: {
    videoPath: string | null
    startTime: number
    duration: number
    videoDuration: number
    ready: boolean
  }) => void
}

const MIN_DURATION = 2

function formatTimecode(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${String(m).padStart(2, '0')}:${s.toFixed(2).padStart(5, '0')}`
}


export function RetakePanel({
  initialVideoPath,
  initialDuration,
  resetKey,
  isProcessing = false,
  processingStatus = '',
  fillHeight = false,
  onChange,
}: RetakePanelProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const filmstripRef = useRef<HTMLDivElement>(null)
  const [videoPath, setVideoPath] = useState<string | null>(initialVideoPath || null)
  const videoUrl = videoPath ? pathToFileUrl(videoPath) : null
  const [videoDuration, setVideoDuration] = useState<number>(initialDuration || 0)

  const [isPlaying, setIsPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [currentTime, setCurrenTime] = useState(0)
  const [isDragOver, setIsDragOver] = useState(false)

  const [selStart, setSelStart] = useState(0)
  const [selEnd, setSelEnd] = useState(Math.min(videoDuration || 0, 5))
  const [draggingHandle, setDraggingHandle] = useState<'start' | 'end' | 'range' | null>(null)
  const dragStartRef = useRef<{ mouseX: number; selStart: number; selEnd: number } | null>(null)
  const initialSelectionAppliedRef = useRef(false)

  const [thumbnails, setThumbnails] = useState<string[]>([])
  const [thumbCount] = useState(20)
  const extractingRef = useRef(false)

  useEffect(() => {
    if (resetKey === undefined) return
    setVideoPath(initialVideoPath || null)
    setVideoDuration(initialDuration || 0)
    setIsPlaying(false)
    setCurrenTime(0)
    setSelStart(0)
    setSelEnd(Math.min(initialDuration || 0, 5))
    setThumbnails([])
    extractingRef.current = false
    initialSelectionAppliedRef.current = false
  }, [resetKey, initialVideoPath, initialDuration])

  useEffect(() => {
    if (!videoUrl) {
      setVideoDuration(0)
      setIsPlaying(false)
      setCurrenTime(0)
      setSelStart(0)
      setSelEnd(0)
      setThumbnails([])
      extractingRef.current = false
      initialSelectionAppliedRef.current = false
      return
    }
    initialSelectionAppliedRef.current = false
  }, [videoUrl, initialDuration])

  useEffect(() => {
    if (!videoUrl || videoDuration <= 0 || initialSelectionAppliedRef.current) return
    setSelStart(0)
    setSelEnd(Math.min(videoDuration, 5))
    initialSelectionAppliedRef.current = true
  }, [videoDuration, videoUrl])

  useEffect(() => {
    const ready = !!videoPath && (selEnd - selStart) >= MIN_DURATION
    onChange?.({
      videoPath,
      startTime: selStart,
      duration: selEnd - selStart,
      videoDuration,
      ready,
    })
  }, [videoUrl, videoPath, selStart, selEnd, videoDuration, onChange])

  useEffect(() => {
    if (!videoUrl || extractingRef.current || videoDuration <= 0) return
    extractingRef.current = true

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
  }, [videoUrl, videoDuration, thumbCount])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const handler = () => setCurrenTime(video.currentTime)
    const onLoaded = () => {
      if ((initialDuration || 0) <= 0 && video.duration && Number.isFinite(video.duration)) {
        setVideoDuration(video.duration)
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

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      const key = e.key.toLowerCase()
      const video = videoRef.current

      if (key === ' ') {
        e.preventDefault()
        e.stopPropagation()
        togglePlay()
      } else if (key === 'i') {
        e.preventDefault()
        e.stopPropagation()
        if (video) {
          const t = video.currentTime
          if (t < selEnd - MIN_DURATION) setSelStart(t)
        }
      } else if (key === 'o') {
        e.preventDefault()
        e.stopPropagation()
        if (video) {
          const t = video.currentTime
          if (t > selStart + MIN_DURATION) setSelEnd(t)
        }
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
      } else if (key === 'j' || key === 'k' || key === 'l') {
        e.preventDefault()
        e.stopPropagation()
        if (key === 'k') {
          if (video) { video.pause(); setIsPlaying(false) }
        } else if (key === 'l') {
          if (video) { video.play(); setIsPlaying(true) }
        }
      }
    }
    window.addEventListener('keydown', handler, true)
    return () => window.removeEventListener('keydown', handler, true)
  }, [togglePlay, selStart, selEnd, videoDuration])

  const selStartRef = useRef(selStart)
  selStartRef.current = selStart
  const selEndRef = useRef(selEnd)
  selEndRef.current = selEnd

  const handleFilmstripMouseDown = useCallback((e: React.MouseEvent, handle: 'start' | 'end' | 'range') => {
    e.preventDefault()
    e.stopPropagation()
    dragStartRef.current = { mouseX: e.clientX, selStart: selStartRef.current, selEnd: selEndRef.current }
    setDraggingHandle(handle)
  }, [])

  useEffect(() => {
    if (!draggingHandle) return

    const handleMouseMove = (e: MouseEvent) => {
      const strip = filmstripRef.current
      const origin = dragStartRef.current
      if (!strip || !origin) return
      const rect = strip.getBoundingClientRect()

      if (draggingHandle === 'range') {
        const dx = e.clientX - origin.mouseX
        const dtSeconds = (dx / rect.width) * videoDuration
        const rangeDuration = origin.selEnd - origin.selStart
        let newStart = origin.selStart + dtSeconds
        let newEnd = origin.selEnd + dtSeconds
        if (newStart < 0) { newStart = 0; newEnd = rangeDuration }
        if (newEnd > videoDuration) { newEnd = videoDuration; newStart = videoDuration - rangeDuration }
        setSelStart(Math.max(0, newStart))
        setSelEnd(Math.min(videoDuration, newEnd))
      } else {
        const fraction = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
        const time = fraction * videoDuration
        if (draggingHandle === 'start') {
          const maxStart = selEndRef.current - MIN_DURATION
          setSelStart(Math.max(0, Math.min(maxStart, time)))
        } else {
          const minEnd = selStartRef.current + MIN_DURATION
          setSelEnd(Math.min(videoDuration, Math.max(minEnd, time)))
        }
      }
    }

    const handleMouseUp = () => {
      setDraggingHandle(null)
      dragStartRef.current = null
    }

    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [draggingHandle, videoDuration])

  const handleFilmstripClick = useCallback((e: React.MouseEvent) => {
    if (draggingHandle) return
    const strip = filmstripRef.current
    const video = videoRef.current
    if (!strip || !video) return
    const rect = strip.getBoundingClientRect()
    const fraction = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    video.currentTime = fraction * videoDuration
  }, [draggingHandle, videoDuration])

  const handleBrowse = useCallback(async () => {
    const paths = await window.electronAPI.showOpenFileDialog({
      title: 'Select Video',
      filters: [{ name: 'Video', extensions: ['mp4', 'mov', 'avi', 'webm', 'mkv'] }],
    })
    if (paths && paths.length > 0) {
      const filePath = paths[0]
      setVideoPath(filePath)
      setThumbnails([])
      extractingRef.current = false
    }
  }, [])

  const handleClear = useCallback(() => {
    setVideoPath(null)
    setVideoDuration(0)
    setIsPlaying(false)
    setCurrenTime(0)
    setSelStart(0)
    setSelEnd(0)
    setThumbnails([])
    extractingRef.current = false
    initialSelectionAppliedRef.current = false
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const assetData = e.dataTransfer.getData('asset')
    if (assetData) {
      try {
        const asset = JSON.parse(assetData) as { type?: string; path?: string }
        if (asset.type === 'video' && asset.path) {
          setVideoPath(asset.path)
          setThumbnails([])
          extractingRef.current = false
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
        setVideoPath(filePath)
        setThumbnails([])
        extractingRef.current = false
      }
    }
  }, [])

  const selStartFrac = videoDuration > 0 ? selStart / videoDuration : 0
  const selEndFrac = videoDuration > 0 ? selEnd / videoDuration : 1
  const playheadFrac = videoDuration > 0 ? currentTime / videoDuration : 0
  const selDuration = selEnd - selStart

  return (
    <div className={`bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden flex flex-col ${fillHeight ? 'h-full min-h-0' : ''}`}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Film className="h-4 w-4 text-blue-400" />
          <span className="text-sm font-semibold text-white">Retake</span>
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
            <p className="text-sm text-white">Drop a video to retake</p>
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

            <div className="px-4 pt-3 pb-1">
              <p className="text-xs font-semibold text-white">Select the video part to regenerate</p>
              <p className="text-[10px] text-zinc-500 mt-0.5">
                Use the prompt panel below to describe what should happen
              </p>
            </div>

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

              <div
                className="absolute top-0 bottom-0 left-0 bg-black/75 pointer-events-none"
                style={{ width: `${selStartFrac * 100}%` }}
              />
              <div
                className="absolute top-0 bottom-0 right-0 bg-black/75 pointer-events-none"
                style={{ width: `${(1 - selEndFrac) * 100}%` }}
              />

              <div
                className="absolute top-0 bottom-0 bg-white pointer-events-none"
                style={{
                  left: `${selStartFrac * 100}%`,
                  width: `${(selEndFrac - selStartFrac) * 100}%`,
                }}
              />

              <div
                className={`absolute top-0 bottom-0 z-[12] ${draggingHandle === 'range' ? 'cursor-grabbing' : 'cursor-grab'}`}
                style={{
                  left: `calc(${selStartFrac * 100}% + 14px)`,
                  width: `calc(${(selEndFrac - selStartFrac) * 100}% - 28px)`,
                }}
                onMouseDown={(e) => handleFilmstripMouseDown(e, 'range')}
              />

              <div
                className="absolute top-0 bottom-0 border-2 border-blue-500 pointer-events-none"
                style={{
                  left: `${selStartFrac * 100}%`,
                  width: `${(selEndFrac - selStartFrac) * 100}%`,
                }}
              />

              <div
                className="absolute top-0 bottom-0 cursor-ew-resize z-20 group"
                style={{ left: `calc(${selStartFrac * 100}% - 6px)`, width: '20px' }}
                onMouseDown={(e) => handleFilmstripMouseDown(e, 'start')}
              >
                <div className="absolute top-0 bottom-0 bg-blue-500 group-hover:bg-blue-400 transition-colors"
                  style={{ left: '5px', width: '4px', borderRadius: '2px 0 0 2px' }}
                />
                <div className="absolute top-0 bg-blue-500 group-hover:bg-blue-400 transition-colors" style={{ left: '5px', width: '10px', height: '3px', borderRadius: '2px 0 0 0' }} />
                <div className="absolute bottom-0 bg-blue-500 group-hover:bg-blue-400 transition-colors" style={{ left: '5px', width: '10px', height: '3px', borderRadius: '0 0 0 2px' }} />
              </div>

              <div
                className="absolute top-0 bottom-0 cursor-ew-resize z-20 group"
                style={{ left: `calc(${selEndFrac * 100}% - 14px)`, width: '20px' }}
                onMouseDown={(e) => handleFilmstripMouseDown(e, 'end')}
              >
                <div className="absolute top-0 bottom-0 bg-blue-500 group-hover:bg-blue-400 transition-colors"
                  style={{ right: '5px', width: '4px', borderRadius: '0 2px 2px 0' }}
                />
                <div className="absolute top-0 bg-blue-500 group-hover:bg-blue-400 transition-colors" style={{ right: '5px', width: '10px', height: '3px', borderRadius: '0 2px 0 0' }} />
                <div className="absolute bottom-0 bg-blue-500 group-hover:bg-blue-400 transition-colors" style={{ right: '5px', width: '10px', height: '3px', borderRadius: '0 0 2px 0' }} />
              </div>

              <div
                className="absolute top-0 bottom-0 w-0.5 bg-zinc-800 pointer-events-none z-[15]"
                style={{ left: `${playheadFrac * 100}%` }}
              />

              <div
                className="absolute top-1/2 pointer-events-none z-10"
                style={{
                  left: `${((selStartFrac + selEndFrac) / 2) * 100}%`,
                  transform: 'translate(-50%, -50%)',
                }}
              >
                <span className="text-[11px] font-mono text-zinc-700 bg-white/90 rounded px-2 py-0.5 font-semibold shadow">
                  {formatTimecode(selDuration)}
                </span>
              </div>
            </div>

              <div className="flex justify-between mt-1.5">
                <span className="text-[10px] font-mono text-blue-400">{formatTimecode(selStart)}</span>
                <span className="text-[10px] font-mono text-zinc-500">Duration: {formatTimecode(selDuration)}</span>
                <span className="text-[10px] font-mono text-blue-400">{formatTimecode(selEnd)}</span>
              </div>
            </div>

            {isProcessing && (
              <div className="px-4 pb-4">
                <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-blue-600/10 border border-blue-500/20">
                  <Loader2 className="h-3.5 w-3.5 text-blue-400 animate-spin flex-shrink-0" />
                  <span className="text-xs text-blue-300">{processingStatus || 'Processing retake...'}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
