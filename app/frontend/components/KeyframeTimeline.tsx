import { useLayoutEffect, useRef, useState } from 'react'
import { RefreshCw, Trash2 } from 'lucide-react'
import { applyTimecode, nudgeKeyframe } from '../lib/keyframe-controls'
import {
  formatKeyframeStrength,
  nudgeKeyframeStrength,
} from '../lib/keyframe-strength'
import {
  findNearestFreeFrameIndex,
  formatKeyframeTimecode,
  frameFromPointer,
  sameDraggedFrame,
  withDraggedFrame,
  type DraggedFrame,
} from '../lib/keyframe-timeline'
import type { KeyframeItem } from '../lib/multi-keyframe'
import { pathToFileUrl } from '../lib/file-url'
import { KeyframeStrengthRail } from './KeyframeStrengthRail'

interface KeyframeTimelineProps {
  keyframes: readonly KeyframeItem[]
  fps: number
  lastFrame: number
  playheadFrame: number
  onPlayheadChange: (frameIndex: number) => void
  onDragFrameChange?: (drag: DraggedFrame | null) => void
  onFrameChange: (id: string, frameIndex: number) => void
  onStrengthChange: (id: string, strength: number) => void
  onReplaceRequest: (id: string) => void
  onDelete: (id: string) => void
  onImagesDrop: (dataTransfer: DataTransfer, replaceId: string | null) => void
}

interface DragState {
  id: string
  frameIndex: number
  pointerId: number
}

export function KeyframeTimeline({
  keyframes,
  fps,
  lastFrame,
  playheadFrame,
  onPlayheadChange,
  onDragFrameChange,
  onFrameChange,
  onStrengthChange,
  onReplaceRequest,
  onDelete,
  onImagesDrop,
}: KeyframeTimelineProps) {
  const trackRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef<DragState | null>(null)
  const onDragFrameChangeRef = useRef(onDragFrameChange)
  onDragFrameChangeRef.current = onDragFrameChange
  const [drag, setDrag] = useState<DragState | null>(null)
  const [timecodeDraft, setTimecodeDraft] = useState<{
    id: string
    value: string
  } | null>(null)

  useLayoutEffect(() => {
    return () => onDragFrameChangeRef.current?.(null)
  }, [])

  const updateDrag = (next: DragState | null) => {
    const prev = dragRef.current
    dragRef.current = next
    const overlay = next ? { id: next.id, frameIndex: next.frameIndex } : null
    const prevOverlay = prev ? { id: prev.id, frameIndex: prev.frameIndex } : null
    if (sameDraggedFrame(prevOverlay, overlay)) return
    setDrag(next)
    onDragFrameChangeRef.current?.(overlay)
  }

  const displayed = withDraggedFrame(
    keyframes,
    drag ? { id: drag.id, frameIndex: drag.frameIndex } : null,
  )

  const frameAtPointer = (clientX: number) => {
    const rect = trackRef.current?.getBoundingClientRect()
    return rect ? frameFromPointer(clientX, rect, lastFrame) : 0
  }

  const handlePointerMove = (event: React.PointerEvent<HTMLElement>) => {
    const activeDrag = dragRef.current
    if (activeDrag && activeDrag.pointerId === event.pointerId) {
      const frameIndex = frameAtPointer(event.clientX)
      updateDrag({ ...activeDrag, frameIndex })
      onPlayheadChange(frameIndex)
      return
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      onPlayheadChange(frameAtPointer(event.clientX))
    }
  }

  const finishDrag = (event: React.PointerEvent<HTMLElement>) => {
    const activeDrag = dragRef.current
    if (!activeDrag || activeDrag.pointerId !== event.pointerId) return
    const otherKeyframes = keyframes.filter((keyframe) => keyframe.id !== activeDrag.id)
    const frameIndex = findNearestFreeFrameIndex(otherKeyframes, frameAtPointer(event.clientX), lastFrame)
    if (frameIndex !== null) onFrameChange(activeDrag.id, frameIndex)
    updateDrag(null)
  }

  const positionPercent = (frameIndex: number) =>
    lastFrame > 0 ? (frameIndex / lastFrame) * 100 : 0

  const commitTimecode = (id: string, value: string) => {
    const frameIndex = applyTimecode(keyframes, id, value, fps, lastFrame)
    if (frameIndex !== null) onFrameChange(id, frameIndex)
    setTimecodeDraft(null)
  }

  return (
    <div>
      <div
        ref={trackRef}
        className="relative h-16 rounded-lg border border-separator bg-surface-tertiary"
        onPointerMove={handlePointerMove}
        onPointerUp={finishDrag}
        onPointerCancel={() => updateDrag(null)}
        onPointerDown={(event) => {
          if (event.target !== event.currentTarget) return
          event.currentTarget.setPointerCapture(event.pointerId)
          onPlayheadChange(frameAtPointer(event.clientX))
        }}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault()
          onImagesDrop(event.dataTransfer, null)
        }}
      >
        <div className="pointer-events-none absolute left-2 right-2 top-1/2 h-px bg-separator" />
        <div
          className="pointer-events-none absolute inset-y-1 w-px bg-blue-400"
          style={{ left: `${positionPercent(playheadFrame)}%` }}
        />

        {displayed.map((keyframe) => {
          const markerFrame = keyframe.frameIndex
          const source = keyframes.find((item) => item.id === keyframe.id) ?? keyframe
          const timecode = formatKeyframeTimecode(markerFrame, fps)
          return (
            <div
              key={keyframe.id}
              className="group absolute top-1/2 z-10 -translate-x-1/2 -translate-y-1/2 touch-none"
              style={{ left: `${positionPercent(markerFrame)}%` }}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault()
                event.stopPropagation()
                onImagesDrop(event.dataTransfer, keyframe.id)
              }}
            >
              <button
                type="button"
                title="Drag to move keyframe"
                aria-label={`Keyframe at ${timecode}`}
                className="relative block h-11 w-11 cursor-ew-resize overflow-hidden rounded-md border-2 border-blue-500 bg-surface-primary shadow-lg"
                onPointerDown={(event) => {
                  event.preventDefault()
                  event.stopPropagation()
                  event.currentTarget.focus({ preventScroll: true })
                  event.currentTarget.setPointerCapture(event.pointerId)
                  updateDrag({
                    id: keyframe.id,
                    frameIndex: source.frameIndex,
                    pointerId: event.pointerId,
                  })
                }}
                onPointerMove={handlePointerMove}
                onPointerUp={finishDrag}
                onPointerCancel={() => updateDrag(null)}
                onKeyDown={(event) => {
                  if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
                    event.preventDefault()
                    event.stopPropagation()
                    onStrengthChange(
                      keyframe.id,
                      nudgeKeyframeStrength(source.strength, event.key === 'ArrowUp' ? 1 : -1),
                    )
                    return
                  }
                  if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
                  event.preventDefault()
                  event.stopPropagation()
                  const frameIndex = nudgeKeyframe(
                    keyframes,
                    keyframe.id,
                    event.key === 'ArrowLeft' ? -1 : 1,
                    lastFrame,
                  )
                  if (frameIndex !== null) onFrameChange(keyframe.id, frameIndex)
                }}
              >
                <img
                  src={pathToFileUrl(source.path)}
                  alt=""
                  draggable={false}
                  className="h-full w-full object-cover"
                />
                <span className="pointer-events-none absolute inset-x-0 bottom-0 bg-[color-mix(in_srgb,var(--semantic-bg-black-primary)_70%,transparent)] py-px text-center text-[9px] font-medium text-fg-white">
                  {formatKeyframeStrength(source.strength)}
                </span>
              </button>
              <KeyframeStrengthRail
                strength={source.strength}
                label={timecode}
                onStrengthChange={(strength) => onStrengthChange(keyframe.id, strength)}
              />
              <div className="absolute -right-3 -top-2 z-20 flex gap-0.5 opacity-0 shadow group-hover:opacity-100 group-focus-within:opacity-100">
                <button
                  type="button"
                  data-keyframe-replace
                  title="Replace keyframe image"
                  aria-label={`Replace keyframe at ${timecode}`}
                  className="rounded-full bg-action p-1 text-fg-secondary hover:text-blue-400"
                  onClick={(event) => {
                    event.stopPropagation()
                    onReplaceRequest(keyframe.id)
                  }}
                >
                  <RefreshCw className="h-3 w-3" />
                </button>
                <button
                  type="button"
                  data-keyframe-delete
                  title="Delete keyframe"
                  aria-label={`Delete keyframe at ${timecode}`}
                  className="rounded-full bg-action p-1 text-fg-secondary hover:text-fg-danger"
                  onClick={(event) => {
                    event.stopPropagation()
                    onDelete(keyframe.id)
                  }}
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
              <input
                type="text"
                aria-label={`Timecode for keyframe at frame ${markerFrame}`}
                className="absolute left-1/2 top-full mt-0.5 h-4 w-[66px] -translate-x-1/2 rounded border border-transparent bg-surface-primary px-1 text-center font-mono text-[9px] text-fg-secondary outline-none hover:border-separator focus:border-blue-500 focus:text-fg-primary"
                value={timecodeDraft?.id === keyframe.id
                  ? timecodeDraft.value
                  : timecode}
                onFocus={(event) => {
                  setTimecodeDraft({ id: keyframe.id, value: event.currentTarget.value })
                }}
                onChange={(event) => {
                  setTimecodeDraft({ id: keyframe.id, value: event.currentTarget.value })
                }}
                onBlur={(event) => commitTimecode(keyframe.id, event.currentTarget.value)}
                onKeyDown={(event) => {
                  event.stopPropagation()
                  if (event.key === 'Enter') event.currentTarget.blur()
                  if (event.key === 'Escape') {
                    event.currentTarget.value = timecode
                    event.currentTarget.blur()
                  }
                }}
              />
            </div>
          )
        })}
      </div>
      <div className="mt-1 flex justify-between font-mono text-[9px] text-fg-tertiary">
        <span>0</span>
        <span>{lastFrame}</span>
      </div>
    </div>
  )
}
