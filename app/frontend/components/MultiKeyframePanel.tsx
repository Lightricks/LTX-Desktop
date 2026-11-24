import { useLayoutEffect, useRef, useState } from 'react'
import { ImagePlus, Plus } from 'lucide-react'
import { imagePathsFromDataTransfer, imagePathsFromFiles } from '../lib/keyframe-drop'
import { lastFrameFromDuration, type DraggedFrame } from '../lib/keyframe-timeline'
import { applyKeyframeImagePaths, type KeyframeItem } from '../lib/multi-keyframe'
import { KeyframeTimeline } from './KeyframeTimeline'

interface MultiKeyframePanelProps {
  keyframes: readonly KeyframeItem[]
  duration: number | null
  fps: number
  maxCount: number
  playheadFrame: number
  onPlayheadChange: (frameIndex: number) => void
  onChange: (keyframes: KeyframeItem[]) => void
  onDragFrameChange?: (drag: DraggedFrame | null) => void
}

export function MultiKeyframePanel({
  keyframes,
  duration,
  fps,
  maxCount,
  playheadFrame,
  onPlayheadChange,
  onChange,
  onDragFrameChange,
}: MultiKeyframePanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const replaceIdRef = useRef<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const lastFrame = duration == null ? null : lastFrameFromDuration(duration, fps)
  const addEnabled = lastFrame !== null && keyframes.length < maxCount

  useLayoutEffect(() => {
    if (lastFrame === null) onDragFrameChange?.(null)
  }, [lastFrame, onDragFrameChange])

  const applyPaths = (paths: string[], replaceId: string | null = null) => {
    onChange(applyKeyframeImagePaths({
      keyframes,
      paths,
      replaceId,
      lastFrame,
      preferredFrame: playheadFrame,
      maxCount,
    }))
  }

  const openFilePicker = (replaceId: string | null) => {
    replaceIdRef.current = replaceId
    fileInputRef.current?.click()
  }

  return (
    <div className="border-b border-separator px-2 pb-2 pt-2">
      <div className="mb-1.5 flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-[11px] font-medium text-fg-secondary">
          <ImagePlus className="h-3.5 w-3.5 text-fg-tertiary" />
          <span>Keyframes</span>
          <span className="text-fg-tertiary">{keyframes.length}/{maxCount}</span>
        </div>
        <button
          type="button"
          disabled={!addEnabled}
          title={keyframes.length >= maxCount
            ? `You can place up to ${maxCount} keyframes`
            : lastFrame === null
              ? 'Choose a duration to place keyframes'
              : 'Add keyframes at the playhead'}
          className="flex h-6 items-center gap-1 rounded-md border border-dashed border-separator px-2 text-[10px] text-fg-secondary hover:border-separator hover:text-fg-primary disabled:cursor-not-allowed disabled:opacity-40"
          onClick={() => openFilePicker(null)}
          onDragOver={(event) => {
            event.preventDefault()
            if (addEnabled) setIsDragOver(true)
          }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={(event) => {
            event.preventDefault()
            setIsDragOver(false)
            applyPaths(imagePathsFromDataTransfer(event.dataTransfer))
          }}
        >
          <Plus className="h-3 w-3" />
          <span className={isDragOver ? 'text-blue-300' : undefined}>Add keyframe</span>
        </button>
      </div>

      {lastFrame === null ? (
        <div className="flex h-12 items-center justify-center rounded-lg border border-dashed border-separator text-[10px] text-fg-tertiary">
          Choose a duration to place keyframes
        </div>
      ) : (
        <KeyframeTimeline
          keyframes={keyframes}
          fps={fps}
          lastFrame={lastFrame}
          playheadFrame={Math.min(playheadFrame, lastFrame)}
          onPlayheadChange={onPlayheadChange}
          onDragFrameChange={onDragFrameChange}
          onFrameChange={(id, frameIndex) => {
            onChange(keyframes.map((keyframe) => (
              keyframe.id === id ? { ...keyframe, frameIndex } : keyframe
            )))
            onPlayheadChange(frameIndex)
          }}
          onStrengthChange={(id, strength) => {
            onChange(keyframes.map((keyframe) => (
              keyframe.id === id ? { ...keyframe, strength } : keyframe
            )))
          }}
          onReplaceRequest={(id) => openFilePicker(id)}
          onDelete={(id) => onChange(keyframes.filter((keyframe) => keyframe.id !== id))}
          onImagesDrop={(dataTransfer, replaceId) => {
            applyPaths(imagePathsFromDataTransfer(dataTransfer), replaceId)
          }}
        />
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={(event) => {
          applyPaths(imagePathsFromFiles(event.target.files ?? []), replaceIdRef.current)
          replaceIdRef.current = null
          event.target.value = ''
        }}
      />
    </div>
  )
}
