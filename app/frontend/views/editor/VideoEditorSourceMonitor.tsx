import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Play, Pause, Square, SkipBack, SkipForward, ChevronLeft, ChevronRight, Repeat, Video, Music, X } from 'lucide-react'
import type { Asset } from '../../types/project-model'
import { formatTime } from './video-editor-utils'
import { Tooltip } from '../../components/ui/tooltip'
import { Text } from '@ds/Text/Text'
import { pathToFileUrl } from '../../lib/file-url'
import { selectHasSourceAsset } from './editor-selectors'
import { useEditorActions, useEditorStore } from './editor-store'

export type SourceKeyboardAction =
  | 'transport.playPause'
  | 'transport.shuttleReverse'
  | 'transport.shuttleStop'
  | 'transport.shuttleForward'
  | 'transport.stepBackward'
  | 'transport.stepForward'
  | 'transport.jumpBackward'
  | 'transport.jumpForward'
  | 'mark.setIn'
  | 'mark.setOut'
  | 'mark.clearIn'
  | 'mark.clearOut'
  | 'mark.clearInOut'
  | 'edit.insertEdit'
  | 'edit.overwriteEdit'

export interface SourceEditRequest {
  asset: Asset
  sourceIn: number | null
  sourceOut: number | null
  sourceTime: number
}

export interface VideoEditorSourceMonitorHandle {
  openAsset: (asset: Asset, opts?: { initialTime?: number; resetMarks?: boolean }) => void
  pause: () => void
  dispatchKeyboardAction: (action: SourceKeyboardAction) => void
}

type SourceMarker = 'sourceIn' | 'sourceOut' | null

const FRAME_DURATION = 1 / 24
const LOOP_EPSILON = 0.01

function getLoopRange(sourceIn: number | null, sourceOut: number | null, duration: number) {
  const start = sourceIn ?? 0
  const end = sourceOut ?? duration

  return start <= end
    ? { start, end }
    : { start: end, end: start }
}

export const VideoEditorSourceMonitor = React.forwardRef<VideoEditorSourceMonitorHandle>(function VideoEditorSourceMonitor(_props, ref) {
  const {
    closeSourceMonitor,
    insertSourceEdit,
    overwriteSourceEdit,
    pause: pauseTimelinePlayback,
    setHasSourceAsset,
    stopShuttle,
  } = useEditorActions()
  const hasSourceAsset = useEditorStore(selectHasSourceAsset)
  const [sourceAsset, setSourceAsset] = useState<Asset | null>(null)
  const [sourceTime, setSourceTime] = useState(0)
  const [sourceIsPlaying, setSourceIsPlaying] = useState(false)
  const [sourceIn, setSourceIn] = useState<number | null>(null)
  const [sourceOut, setSourceOut] = useState<number | null>(null)
  const [sourcePlayingInOut, setSourcePlayingInOut] = useState(false)
  const [sourceReversePlaying, setSourceReversePlaying] = useState(false)
  const [draggingMarker, setDraggingMarker] = useState<SourceMarker>(null)

  const sourceVideoRef = useRef<HTMLVideoElement>(null)
  const sourceAnimRef = useRef<number>(0)
  const reverseRafRef = useRef<number | null>(null)
  const reverseLastRef = useRef<number | null>(null)
  const pendingSeekRef = useRef<number | null>(null)
  const sourceTimeRef = useRef(0)
  sourceTimeRef.current = sourceTime
  const sourceInRef = useRef<number | null>(null)
  sourceInRef.current = sourceIn
  const sourceOutRef = useRef<number | null>(null)
  sourceOutRef.current = sourceOut

  const pause = useCallback(() => {
    sourceVideoRef.current?.pause()
    setSourceIsPlaying(false)
    setSourceReversePlaying(false)
  }, [])

  const stopPlayback = useCallback(() => {
    sourceVideoRef.current?.pause()
    setSourceIsPlaying(false)
    setSourcePlayingInOut(false)
    setSourceReversePlaying(false)
  }, [])

  const seekTo = useCallback((t: number) => {
    setSourceTime(t)
    if (sourceVideoRef.current) sourceVideoRef.current.currentTime = t
  }, [])

  const emitInsertRequest = useCallback(() => {
    if (!sourceAsset) return
    insertSourceEdit({ asset: sourceAsset, sourceIn, sourceOut, sourceTime })
  }, [insertSourceEdit, sourceAsset, sourceIn, sourceOut, sourceTime])

  const emitOverwriteRequest = useCallback(() => {
    if (!sourceAsset) return
    overwriteSourceEdit({ asset: sourceAsset, sourceIn, sourceOut, sourceTime })
  }, [overwriteSourceEdit, sourceAsset, sourceIn, sourceOut, sourceTime])

  const dispatchKeyboardAction = useCallback((action: SourceKeyboardAction) => {
    switch (action) {
      case 'transport.playPause':
        setSourceReversePlaying(false)
        if (sourceIsPlaying) {
          pause()
        } else {
          pauseTimelinePlayback()
          stopShuttle()
          if (sourceVideoRef.current) {
            const duration = sourceAsset?.duration || 5
            const { start: loopStart, end: loopEnd } = getLoopRange(sourceIn, sourceOut, duration)
            const looping = sourcePlayingInOut
            if (looping) {
              if (sourceTimeRef.current < loopStart || sourceTimeRef.current >= loopEnd - LOOP_EPSILON) {
                sourceVideoRef.current.currentTime = loopStart
                setSourceTime(loopStart)
              }
            } else if (sourceIn !== null && sourceTimeRef.current < sourceIn) {
              sourceVideoRef.current.currentTime = sourceIn
              setSourceTime(sourceIn)
            }
            sourceVideoRef.current.play().catch(() => {})
          }
          setSourceIsPlaying(true)
        }
        return
      case 'transport.shuttleReverse': {
        setSourceReversePlaying(false)
        const video = sourceVideoRef.current
        if (!video) return
        video.pause()
        setSourceIsPlaying(false)
        const t = Math.max(0, video.currentTime - FRAME_DURATION)
        video.currentTime = t
        setSourceTime(t)
        return
      }
      case 'transport.shuttleStop':
        stopPlayback()
        return
      case 'transport.shuttleForward':
        setSourceReversePlaying(false)
        pauseTimelinePlayback()
        stopShuttle()
        if (sourceVideoRef.current) sourceVideoRef.current.play().catch(() => {})
        setSourceIsPlaying(true)
        return
      case 'transport.stepBackward': {
        setSourceReversePlaying(false)
        const t = Math.max(0, sourceTimeRef.current - FRAME_DURATION)
        seekTo(t)
        return
      }
      case 'transport.stepForward': {
        setSourceReversePlaying(false)
        const dur = sourceAsset?.duration || 5
        const t = Math.min(dur, sourceTimeRef.current + FRAME_DURATION)
        seekTo(t)
        return
      }
      case 'transport.jumpBackward': {
        setSourceReversePlaying(false)
        const t = Math.max(0, sourceTimeRef.current - 1)
        seekTo(t)
        return
      }
      case 'transport.jumpForward': {
        setSourceReversePlaying(false)
        const dur = sourceAsset?.duration || 5
        const t = Math.min(dur, sourceTimeRef.current + 1)
        seekTo(t)
        return
      }
      case 'mark.setIn': {
        const st = sourceTimeRef.current
        setSourceIn(prev => prev !== null && Math.abs(prev - st) < 0.01 ? null : st)
        return
      }
      case 'mark.setOut': {
        const st = sourceTimeRef.current
        setSourceOut(prev => prev !== null && Math.abs(prev - st) < 0.01 ? null : st)
        return
      }
      case 'mark.clearIn':
        setSourceIn(null)
        return
      case 'mark.clearOut':
        setSourceOut(null)
        return
      case 'mark.clearInOut':
        setSourceIn(null)
        setSourceOut(null)
        return
      case 'edit.insertEdit':
        emitInsertRequest()
        return
      case 'edit.overwriteEdit':
        emitOverwriteRequest()
        return
      default:
        return
    }
  }, [
    emitInsertRequest,
    emitOverwriteRequest,
    pause,
    pauseTimelinePlayback,
    seekTo,
    sourceAsset,
    sourceIn,
    sourceIsPlaying,
    sourceOut,
    sourcePlayingInOut,
    stopPlayback,
    stopShuttle,
  ])

  const openAsset = useCallback((asset: Asset, opts?: { initialTime?: number; resetMarks?: boolean }) => {
    const nextTime = Math.max(0, opts?.initialTime ?? 0)
    const shouldResetMarks = opts?.resetMarks ?? true

    stopPlayback()
    setSourceAsset(asset)
    setSourceTime(nextTime)
    if (shouldResetMarks) {
      setSourceIn(null)
      setSourceOut(null)
    }
    pendingSeekRef.current = nextTime
  }, [stopPlayback])

  React.useImperativeHandle(ref, () => ({
    openAsset,
    pause,
    dispatchKeyboardAction,
  }), [dispatchKeyboardAction, openAsset, pause])

  useEffect(() => {
    const hasAsset = !!sourceAsset
    if (hasSourceAsset === hasAsset) return
    setHasSourceAsset(hasAsset)
  }, [hasSourceAsset, setHasSourceAsset, sourceAsset])

  useEffect(() => () => {
    setHasSourceAsset(false)
  }, [setHasSourceAsset])

  useEffect(() => {
    if (!sourceAsset || sourceAsset.type !== 'video') return
    const pending = pendingSeekRef.current
    if (pending === null || !sourceVideoRef.current) return
    sourceVideoRef.current.currentTime = pending
    pendingSeekRef.current = null
  }, [sourceAsset])

  useEffect(() => {
    if (!sourceIsPlaying || !sourceVideoRef.current) {
      cancelAnimationFrame(sourceAnimRef.current)
      return
    }
    const video = sourceVideoRef.current
    const duration = sourceAsset?.duration || 5
    const { start: loopStart, end: loopEnd } = getLoopRange(sourceIn, sourceOut, duration)
    const tick = () => {
      setSourceTime(video.currentTime)
      if (sourcePlayingInOut && video.currentTime >= loopEnd - LOOP_EPSILON) {
        video.currentTime = loopStart
        setSourceTime(loopStart)
        video.play().catch(() => {})
      } else if (!sourcePlayingInOut && sourceOut !== null && video.currentTime >= sourceOut) {
        video.pause()
        setSourceIsPlaying(false)
        setSourceTime(sourceOut)
        return
      }
      if (sourcePlayingInOut || !video.paused) sourceAnimRef.current = requestAnimationFrame(tick)
    }
    video.play().catch(() => {})
    sourceAnimRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(sourceAnimRef.current)
  }, [sourceAsset, sourceIn, sourceIsPlaying, sourceOut, sourcePlayingInOut])

  useEffect(() => {
    if (!sourceReversePlaying) {
      if (reverseRafRef.current) cancelAnimationFrame(reverseRafRef.current)
      reverseRafRef.current = null
      reverseLastRef.current = null
      return
    }
    sourceVideoRef.current?.pause()
    const tick = (ts: number) => {
      if (!sourceReversePlaying) return
      if (reverseLastRef.current !== null) {
        const delta = (ts - reverseLastRef.current) / 1000
        const duration = sourceAsset?.duration || 5
        const { start: loopStart, end: loopEnd } = getLoopRange(sourceInRef.current, sourceOutRef.current, duration)
        let next = Math.max(0, (sourceVideoRef.current?.currentTime ?? sourceTimeRef.current) - delta)
        if (sourcePlayingInOut && next <= loopStart + LOOP_EPSILON) {
          next = loopEnd
        }
        if (sourceVideoRef.current) sourceVideoRef.current.currentTime = next
        setSourceTime(next)
        if (!sourcePlayingInOut && next <= 0) {
          setSourceReversePlaying(false)
          return
        }
      }
      reverseLastRef.current = ts
      reverseRafRef.current = requestAnimationFrame(tick)
    }
    reverseRafRef.current = requestAnimationFrame(tick)
    return () => {
      if (reverseRafRef.current) cancelAnimationFrame(reverseRafRef.current)
    }
  }, [sourceAsset, sourcePlayingInOut, sourceReversePlaying])

  useEffect(() => {
    if (!draggingMarker) return
    const handleMouseMove = (e: MouseEvent) => {
      const scrubEl = document.getElementById('source-scrub-bar')
      if (!scrubEl) return
      const rect = scrubEl.getBoundingClientRect()
      const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
      const dur = sourceAsset?.duration || 5
      let time = pct * dur
      if (draggingMarker === 'sourceIn' && sourceOutRef.current !== null) {
        time = Math.min(time, sourceOutRef.current - 0.01)
      }
      if (draggingMarker === 'sourceOut' && sourceInRef.current !== null) {
        time = Math.max(time, sourceInRef.current + 0.01)
      }
      time = Math.max(0, Math.min(time, dur))
      if (draggingMarker === 'sourceIn') {
        setSourceIn(time)
      } else {
        setSourceOut(time)
      }
    }
    const handleMouseUp = () => setDraggingMarker(null)
    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [draggingMarker, sourceAsset])

  return (
    <div className="flex flex-col flex-1 min-w-0 min-h-0">
      <div className="h-7 bg-surface-primary border-b border-separator flex items-center justify-between px-3 flex-shrink-0">
        <Text as="span" variant="label" size="sm" className="tracking-wide text-fg-secondary">
          Clip Viewer
        </Text>
        <Tooltip content="Close clip viewer" side="left">
          <button
            onClick={() => {
              stopPlayback()
              closeSourceMonitor()
            }}
            className="text-fg-tertiary hover:text-fg-primary"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </Tooltip>
      </div>

      <div className="flex-1 relative overflow-hidden bg-black flex items-center justify-center min-h-0">
        {sourceAsset ? (
          <>
            {sourceAsset.type === 'video' ? (
              <video
                ref={sourceVideoRef}
                src={pathToFileUrl(sourceAsset.path)}
                className="max-w-full max-h-full object-contain"
                onLoadedMetadata={() => {
                  const pending = pendingSeekRef.current
                  if (pending !== null && sourceVideoRef.current) {
                    sourceVideoRef.current.currentTime = pending
                    pendingSeekRef.current = null
                  }
                }}
                onTimeUpdate={() => {
                  if (sourceVideoRef.current) setSourceTime(sourceVideoRef.current.currentTime)
                }}
                onEnded={() => {
                  if (!sourcePlayingInOut || !sourceVideoRef.current) {
                    setSourceIsPlaying(false)
                    return
                  }
                  const { start: loopStart } = getLoopRange(sourceIn, sourceOut, sourceAsset.duration || 5)
                  sourceVideoRef.current.currentTime = loopStart
                  setSourceTime(loopStart)
                  sourceVideoRef.current.play().catch(() => {})
                  setSourceIsPlaying(true)
                }}
                playsInline
              />
            ) : sourceAsset.type === 'image' ? (
              <img src={pathToFileUrl(sourceAsset.path)} alt="" className="max-w-full max-h-full object-contain" />
            ) : (
              <div className="text-center text-fg-tertiary">
                <Music className="h-12 w-12 mx-auto mb-2" />
                <p className="text-sm">{sourceAsset.path?.split('/').pop() || 'Audio'}</p>
              </div>
            )}
          </>
        ) : (
          <div className="text-center text-fg-tertiary">
            <Video className="h-10 w-10 mx-auto mb-2" />
            <p className="text-xs">Double-click an asset to load it here</p>
          </div>
        )}
      </div>

      {sourceAsset && (sourceAsset.type === 'video' || sourceAsset.type === 'audio') && (
        <div className="bg-surface-primary border-t border-separator flex-shrink-0 relative px-2 py-1">
          <div
            id="source-scrub-bar"
            className="relative h-6 cursor-pointer group"
            onMouseDown={(e) => {
              const bar = e.currentTarget
              const rect = bar.getBoundingClientRect()
              const dur = sourceAsset.duration || 5
              const seek = (clientX: number) => {
                const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
                const t = frac * dur
                seekTo(t)
              }
              seek(e.clientX)
              const onMove = (ev: MouseEvent) => seek(ev.clientX)
              const onUp = () => {
                window.removeEventListener('mousemove', onMove)
                window.removeEventListener('mouseup', onUp)
              }
              window.addEventListener('mousemove', onMove)
              window.addEventListener('mouseup', onUp)
            }}
          >
            <div className="absolute left-0 right-0 top-1/2 -translate-y-1/2 h-1 bg-action-hover rounded-full" />
            {sourceIn !== null && (
              <div
                className="absolute top-0 bottom-0 left-0 bg-black/50 rounded-l"
                style={{ width: `${(sourceIn / (sourceAsset.duration || 5)) * 100}%` }}
              />
            )}
            {sourceOut !== null && (
              <div
                className="absolute top-0 bottom-0 right-0 bg-black/50 rounded-r"
                style={{ width: `${100 - (sourceOut / (sourceAsset.duration || 5)) * 100}%` }}
              />
            )}
            {(sourceIn !== null || sourceOut !== null) && (
              <div
                className="absolute top-0 bottom-0 border-t-2 border-b-2 border-blue-400/70"
                style={{
                  left: `${((sourceIn ?? 0) / (sourceAsset.duration || 5)) * 100}%`,
                  width: `${(((sourceOut ?? sourceAsset.duration ?? 5) - (sourceIn ?? 0)) / (sourceAsset.duration || 5)) * 100}%`,
                }}
              >
                <div className="absolute inset-0 top-1/2 -translate-y-1/2 h-1 bg-blue-400/40 rounded-full" />
              </div>
            )}
            {sourceIn !== null && (
              <div
                className="absolute top-0 bottom-0 flex items-center z-10 cursor-ew-resize"
                style={{ left: `calc(${(sourceIn / (sourceAsset.duration || 5)) * 100}% - 8px)`, width: 14 }}
                onMouseDown={(e) => { e.stopPropagation(); e.preventDefault(); setDraggingMarker('sourceIn') }}
              >
                <div className="w-1.5 h-full bg-blue-400 rounded-l-sm flex flex-col justify-between py-0.5 pointer-events-none ml-auto">
                  <div className="w-2.5 h-0.5 bg-blue-400 rounded-r" />
                  <div className="w-2.5 h-0.5 bg-blue-400 rounded-r" />
                </div>
              </div>
            )}
            {sourceOut !== null && (
              <div
                className="absolute top-0 bottom-0 flex items-center z-10 cursor-ew-resize"
                style={{ left: `${(sourceOut / (sourceAsset.duration || 5)) * 100}%`, width: 14 }}
                onMouseDown={(e) => { e.stopPropagation(); e.preventDefault(); setDraggingMarker('sourceOut') }}
              >
                <div className="w-1.5 h-full bg-blue-400 rounded-r-sm flex flex-col justify-between py-0.5 pointer-events-none">
                  <div className="w-2.5 h-0.5 bg-blue-400 rounded-l -ml-1" />
                  <div className="w-2.5 h-0.5 bg-blue-400 rounded-l -ml-1" />
                </div>
              </div>
            )}
            <div
              className="absolute top-0 bottom-0 z-20"
              style={{ left: `${(sourceTime / (sourceAsset.duration || 5)) * 100}%` }}
            >
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-2 h-2 bg-blue-400 clip-triangle" style={{ clipPath: 'polygon(50% 100%, 0% 0%, 100% 0%)' }} />
              <div className="absolute top-2 bottom-0 left-1/2 -translate-x-1/2 w-px bg-blue-400" />
            </div>
          </div>

          {(sourceIn !== null || sourceOut !== null) && (
            <div className="flex justify-between items-center mt-0.5 h-3.5">
              <span className="text-[9px] font-mono text-blue-400/80">
                {sourceIn !== null ? `IN ${formatTime(sourceIn)}` : ''}
              </span>
              <span className="text-[9px] font-mono text-fg-tertiary">
                {sourceIn !== null && sourceOut !== null
                  ? `Duration: ${formatTime(sourceOut - sourceIn)}`
                  : ''}
              </span>
              <span className="text-[9px] font-mono text-blue-400/80">
                {sourceOut !== null ? `OUT ${formatTime(sourceOut)}` : ''}
              </span>
            </div>
          )}
        </div>
      )}

      <div className="h-8 bg-surface-secondary border-t border-separator flex items-center px-3 flex-shrink-0 gap-2">
        <span className="text-[12px] font-mono font-medium text-fg-primary tabular-nums tracking-tight select-none min-w-[90px]">
          {formatTime(sourceTime)}
        </span>
        <div className="flex-1 flex items-center justify-center gap-0.5">
          <Tooltip content={sourceIn !== null ? `In: ${formatTime(sourceIn)}` : 'Set In (I)'} side="top">
            <button
              onClick={() => setSourceIn(prev => prev !== null && Math.abs(prev - sourceTime) < 0.01 ? null : sourceTime)}
              className={`h-6 w-6 flex items-center justify-center rounded transition-colors ${sourceIn !== null ? 'text-fg-warning' : 'text-fg-tertiary hover:text-fg-primary hover:bg-action'}`}
            >
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="7,4 4,4 4,20 7,20" />
                <line x1="10" y1="12" x2="20" y2="12" />
                <polyline points="16,8 20,12 16,16" />
              </svg>
            </button>
          </Tooltip>
          <div className="w-px h-3 bg-action-hover" />
          <Tooltip content="Go to start" side="top">
            <button
              onClick={() => seekTo(sourceIn ?? 0)}
              className="h-6 w-6 flex items-center justify-center rounded text-fg-tertiary hover:text-fg-primary hover:bg-action transition-colors"
            >
              <SkipBack className="h-3 w-3" />
            </button>
          </Tooltip>
          <Tooltip content="Step back" side="top">
            <button
              onClick={() => dispatchKeyboardAction('transport.stepBackward')}
              className="h-6 w-6 flex items-center justify-center rounded text-fg-tertiary hover:text-fg-primary hover:bg-action transition-colors"
            >
              <ChevronLeft className="h-3.5 w-3.5" />
            </button>
          </Tooltip>
          <Tooltip content="Play reverse" side="top">
            <button
              onClick={() => {
                if (sourceReversePlaying) {
                  setSourceReversePlaying(false)
                } else {
                  sourceVideoRef.current?.pause()
                  setSourceIsPlaying(false)
                  setSourceReversePlaying(true)
                }
              }}
              className={`h-6 w-6 flex items-center justify-center rounded transition-colors ${sourceReversePlaying ? 'text-blue-400' : 'text-fg-tertiary hover:text-fg-primary hover:bg-action'}`}
            >
              <Play className="h-3 w-3 mr-0.5 rotate-180" />
            </button>
          </Tooltip>
          <Tooltip content="Stop" side="top">
            <button
              onClick={() => dispatchKeyboardAction('transport.shuttleStop')}
              className="h-6 w-6 flex items-center justify-center rounded text-fg-tertiary hover:text-fg-primary hover:bg-action transition-colors"
            >
              <Square className="h-2.5 w-2.5" />
            </button>
          </Tooltip>
          <Tooltip content={sourceIsPlaying ? 'Pause' : 'Play'} side="top">
            <button
              onClick={() => dispatchKeyboardAction('transport.playPause')}
              className="h-6 w-6 flex items-center justify-center rounded text-fg-secondary hover:text-fg-primary hover:bg-action transition-colors"
            >
              {sourceIsPlaying ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3 ml-0.5" />}
            </button>
          </Tooltip>
          <Tooltip content="Step forward" side="top">
            <button
              onClick={() => dispatchKeyboardAction('transport.stepForward')}
              className="h-6 w-6 flex items-center justify-center rounded text-fg-tertiary hover:text-fg-primary hover:bg-action transition-colors"
            >
              <ChevronRight className="h-3.5 w-3.5" />
            </button>
          </Tooltip>
          <Tooltip content="Go to end" side="top">
            <button
              onClick={() => seekTo(sourceOut ?? (sourceAsset?.duration || 5))}
              className="h-6 w-6 flex items-center justify-center rounded text-fg-tertiary hover:text-fg-primary hover:bg-action transition-colors"
            >
              <SkipForward className="h-3 w-3" />
            </button>
          </Tooltip>
          <div className="w-px h-3 bg-action-hover" />
          <Tooltip content={sourceOut !== null ? `Out: ${formatTime(sourceOut)}` : 'Set Out (O)'} side="top">
            <button
              onClick={() => setSourceOut(prev => prev !== null && Math.abs(prev - sourceTime) < 0.01 ? null : sourceTime)}
              className={`h-6 w-6 flex items-center justify-center rounded transition-colors ${sourceOut !== null ? 'text-fg-warning' : 'text-fg-tertiary hover:text-fg-primary hover:bg-action'}`}
            >
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="17,4 20,4 20,20 17,20" />
                <line x1="14" y1="12" x2="4" y2="12" />
                <polyline points="8,8 4,12 8,16" />
              </svg>
            </button>
          </Tooltip>
          <Tooltip content="Loop In/Out" side="top">
            <button
              onClick={() => {
                if (sourcePlayingInOut) {
                  stopPlayback()
                  return
                }
                const loopStart = sourceIn ?? 0
                pauseTimelinePlayback()
                stopShuttle()
                setSourceReversePlaying(false)
                setSourcePlayingInOut(true)
                seekTo(loopStart)
                if (sourceVideoRef.current) {
                  sourceVideoRef.current.currentTime = loopStart
                  sourceVideoRef.current.play().catch(() => {})
                }
                setSourceIsPlaying(true)
              }}
              className={`h-6 w-6 flex items-center justify-center rounded transition-colors ${
                sourcePlayingInOut ? 'text-fg-warning bg-warning-soft' : 'text-fg-tertiary hover:text-fg-primary hover:bg-action'
              }`}
            >
              <Repeat className="h-3 w-3" />
            </button>
          </Tooltip>
          <div className="w-px h-3 bg-action-hover mx-0.5" />
          <Tooltip content="Insert Edit (,)" side="top">
            <button
              onClick={emitInsertRequest}
              disabled={!sourceAsset}
              className="h-6 px-1 flex items-center rounded text-fg-secondary hover:text-fg-primary hover:bg-action disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M12 5v14M5 12h14" /></svg>
            </button>
          </Tooltip>
          <Tooltip content="Overwrite Edit (.)" side="top">
            <button
              onClick={emitOverwriteRequest}
              disabled={!sourceAsset}
              className="h-6 px-1 flex items-center rounded text-fg-secondary hover:text-fg-primary hover:bg-action disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M9 12h6" /></svg>
            </button>
          </Tooltip>
        </div>
        <span className="text-[12px] font-mono font-medium text-fg-secondary tabular-nums tracking-tight select-none min-w-[90px] text-right">
          {formatTime(sourceAsset?.duration || 0)}
        </span>
      </div>
    </div>
  )
})
