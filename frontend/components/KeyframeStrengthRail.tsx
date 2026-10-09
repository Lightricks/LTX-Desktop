import { useRef } from 'react'
import {
  formatKeyframeStrength,
  nudgeKeyframeStrength,
  strengthFromPointer,
} from '../lib/keyframe-strength'

interface KeyframeStrengthRailProps {
  strength: number
  label: string
  onStrengthChange: (strength: number) => void
}

export function KeyframeStrengthRail({
  strength,
  label,
  onStrengthChange,
}: KeyframeStrengthRailProps) {
  const railRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef(false)
  const clamped = Math.min(1, Math.max(0, strength))
  const percent = formatKeyframeStrength(strength)

  const applyFromClientY = (clientY: number) => {
    const rect = railRef.current?.getBoundingClientRect()
    if (!rect) return
    onStrengthChange(strengthFromPointer(clientY, rect))
  }

  return (
    <div
      ref={railRef}
      role="slider"
      aria-orientation="vertical"
      tabIndex={0}
      data-keyframe-strength
      aria-label={`Strength for keyframe at ${label}`}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(clamped * 100)}
      aria-valuetext={percent}
      title={`Strength ${percent}`}
      className="absolute left-0 top-0 z-20 h-11 w-3 cursor-ns-resize touch-none"
      onPointerDown={(event) => {
        event.preventDefault()
        event.stopPropagation()
        draggingRef.current = true
        event.currentTarget.setPointerCapture(event.pointerId)
        applyFromClientY(event.clientY)
      }}
      onPointerMove={(event) => {
        if (!draggingRef.current) return
        event.stopPropagation()
        applyFromClientY(event.clientY)
      }}
      onPointerUp={(event) => {
        draggingRef.current = false
        event.stopPropagation()
      }}
      onPointerCancel={() => {
        draggingRef.current = false
      }}
      onLostPointerCapture={() => {
        draggingRef.current = false
      }}
      onKeyDown={(event) => {
        if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return
        event.preventDefault()
        event.stopPropagation()
        onStrengthChange(nudgeKeyframeStrength(strength, event.key === 'ArrowUp' ? 1 : -1))
      }}
    >
      <div className="relative mx-auto h-full w-0.5 rounded-full bg-separator">
        <div
          className="absolute bottom-0 w-full rounded-full bg-brand"
          style={{ height: `${clamped * 100}%` }}
        />
        <div
          className="absolute left-1/2 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-fg-primary shadow"
          style={{ top: `${(1 - clamped) * 100}%` }}
        />
      </div>
    </div>
  )
}
