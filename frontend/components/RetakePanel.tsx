import React, { useState, useRef, useEffect, useCallback } from 'react'
import { VideoPreviewPanel, formatTimecode, type VideoPreviewContext } from './VideoPreviewPanel'
import { validateVideoSource } from '../lib/video-constraints'

interface RetakePanelProps {
  initialVideoPath?: string | null
  initialDuration?: number
  resetKey?: number
  isProcessing?: boolean
  processingStatus?: string
  fillHeight?: boolean
  enforceApiConstraints?: boolean
  onChange?: (data: {
    videoPath: string | null
    startTime: number
    duration: number
    videoDuration: number
    width: number
    height: number
    ready: boolean
  }) => void
}

const MIN_DURATION = 2

export function RetakePanel({
  initialVideoPath,
  initialDuration,
  resetKey,
  isProcessing = false,
  processingStatus = '',
  fillHeight = false,
  enforceApiConstraints = true,
  onChange,
}: RetakePanelProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const filmstripRef = useRef<HTMLDivElement>(null)

  const [videoPath, setVideoPath] = useState<string | null>(initialVideoPath || null)
  const [videoDuration, setVideoDuration] = useState<number>(initialDuration || 0)
  const [dimensions, setDimensions] = useState<{ width: number; height: number }>({ width: 0, height: 0 })

  const [selStart, setSelStart] = useState(0)
  const [selEnd, setSelEnd] = useState(Math.min(initialDuration || 0, 5))
  const [draggingHandle, setDraggingHandle] = useState<'start' | 'end' | 'range' | null>(null)
  // Read synchronously by the filmstrip click guard. Stays true through the click the
  // browser fires right after a drag-release mouseup (cleared on the next tick), so
  // releasing a trim drag over the strip doesn't jump the playhead.
  const draggingRef = useRef(false)
  const dragStartRef = useRef<{ mouseX: number; selStart: number; selEnd: number } | null>(null)
  const initialSelectionAppliedRef = useRef(false)

  useEffect(() => {
    if (resetKey === undefined) return
    setSelStart(0)
    setSelEnd(Math.min(initialDuration || 0, 5))
    initialSelectionAppliedRef.current = false
  }, [resetKey, initialDuration])

  const handleSourceChange = useCallback((data: { videoPath: string | null; videoDuration: number; width: number; height: number }) => {
    setVideoPath(data.videoPath)
    setVideoDuration(data.videoDuration)
    setDimensions({ width: data.width, height: data.height })
    if (!data.videoPath) {
      setSelStart(0)
      setSelEnd(0)
      initialSelectionAppliedRef.current = false
    }
  }, [])

  const sourceError = enforceApiConstraints && videoPath
    ? validateVideoSource({ width: dimensions.width, height: dimensions.height, duration: videoDuration })
    : null

  // Apply the default selection once the source duration is known.
  useEffect(() => {
    if (!videoPath || videoDuration <= 0 || initialSelectionAppliedRef.current) return
    setSelStart(0)
    setSelEnd(Math.min(videoDuration, 5))
    initialSelectionAppliedRef.current = true
  }, [videoDuration, videoPath])

  useEffect(() => {
    const ready = !!videoPath && (selEnd - selStart) >= MIN_DURATION && !sourceError
    onChange?.({
      videoPath,
      startTime: selStart,
      duration: selEnd - selStart,
      videoDuration,
      width: dimensions.width,
      height: dimensions.height,
      ready,
    })
  }, [videoPath, selStart, selEnd, videoDuration, dimensions, sourceError, onChange])

  // Retake-specific keyboard: set in/out points to the playhead.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      const key = e.key.toLowerCase()
      const video = videoRef.current
      if (!video) return
      if (key === 'i') {
        e.preventDefault()
        e.stopPropagation()
        const t = video.currentTime
        if (t < selEnd - MIN_DURATION) setSelStart(t)
      } else if (key === 'o') {
        e.preventDefault()
        e.stopPropagation()
        const t = video.currentTime
        if (t > selStart + MIN_DURATION) setSelEnd(t)
      }
    }
    window.addEventListener('keydown', handler, true)
    return () => window.removeEventListener('keydown', handler, true)
  }, [selStart, selEnd])

  const selStartRef = useRef(selStart)
  selStartRef.current = selStart
  const selEndRef = useRef(selEnd)
  selEndRef.current = selEnd

  const handleFilmstripMouseDown = useCallback((e: React.MouseEvent, handle: 'start' | 'end' | 'range') => {
    e.preventDefault()
    e.stopPropagation()
    dragStartRef.current = { mouseX: e.clientX, selStart: selStartRef.current, selEnd: selEndRef.current }
    draggingRef.current = true
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
      // Clear on the next tick so the click synthesized from this mouseup still sees the
      // drag in progress and is suppressed by the filmstrip seek guard.
      setTimeout(() => { draggingRef.current = false }, 0)
    }

    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [draggingHandle, videoDuration])

  const renderTrimOverlay = useCallback((ctx: VideoPreviewContext) => {
    const dur = ctx.videoDuration
    const selStartFrac = dur > 0 ? selStart / dur : 0
    const selEndFrac = dur > 0 ? selEnd / dur : 1
    return (
      <>
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
          style={{ left: `${selStartFrac * 100}%`, width: `${(selEndFrac - selStartFrac) * 100}%` }}
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
          style={{ left: `${selStartFrac * 100}%`, width: `${(selEndFrac - selStartFrac) * 100}%` }}
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
          className="absolute top-1/2 pointer-events-none z-10"
          style={{ left: `${((selStartFrac + selEndFrac) / 2) * 100}%`, transform: 'translate(-50%, -50%)' }}
        >
          <span className="text-[11px] font-mono text-zinc-700 bg-white/90 rounded px-2 py-0.5 font-semibold shadow">
            {formatTimecode(selEnd - selStart)}
          </span>
        </div>
      </>
    )
  }, [selStart, selEnd, draggingHandle, handleFilmstripMouseDown])

  const renderLabels = useCallback(() => (
    <div className="flex justify-between mt-1.5">
      <span className="text-[10px] font-mono text-blue-400">{formatTimecode(selStart)}</span>
      <span className="text-[10px] font-mono text-zinc-500">Duration: {formatTimecode(selEnd - selStart)}</span>
      <span className="text-[10px] font-mono text-blue-400">{formatTimecode(selEnd)}</span>
    </div>
  ), [selStart, selEnd])

  return (
    <VideoPreviewPanel
      title="Retake"
      initialVideoPath={initialVideoPath}
      initialDuration={initialDuration}
      resetKey={resetKey}
      isProcessing={isProcessing}
      processingStatus={processingStatus}
      processingDefault="Processing retake..."
      fillHeight={fillHeight}
      emptyTitle="Drop a video to retake"
      hint={{ title: 'Select the video part to regenerate', subtitle: 'Use the prompt panel below to describe what should happen' }}
      errorMessage={sourceError ?? undefined}
      videoRef={videoRef}
      filmstripRef={filmstripRef}
      onSourceChange={handleSourceChange}
      shouldSuppressSeek={() => draggingRef.current}
      filmstripOverlay={renderTrimOverlay}
      belowFilmstrip={renderLabels}
    />
  )
}
