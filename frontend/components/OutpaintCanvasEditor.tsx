import { useCallback } from 'react'

export interface OutpaintPads {
  left: number
  right: number
  top: number
  bottom: number
}

interface OutpaintCanvasEditorProps {
  // Source video dimensions in pixels (the silhouette, fixed — never resized).
  sourceWidth: number
  sourceHeight: number
  value: OutpaintPads
  onChange: (pads: OutpaintPads) => void
}

// Square display box for the editor. The source sits centred; the output frame grows around it.
const STAGE = 260
// Aspect-ratio quick-snaps: expand the deficient axis to hit the ratio, centred. null = reset.
const PRESETS: { id: string; label: string; ratio: number | null }[] = [
  { id: 'reset', label: 'Reset', ratio: null },
  { id: '16:9', label: '16:9', ratio: 16 / 9 },
  { id: '1:1', label: '1:1', ratio: 1 },
  { id: '9:16', label: '9:16', ratio: 9 / 16 },
  { id: '21:9', label: '21:9', ratio: 21 / 9 },
]

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))
const toEven = (v: number) => v + (v % 2)

// Handles: each names the edge(s) it moves ('l'/'r'/'t'/'b'). Corners move two.
const HANDLES = ['tl', 't', 'tr', 'l', 'r', 'bl', 'b', 'br'] as const
type Handle = (typeof HANDLES)[number]

export function OutpaintCanvasEditor({ sourceWidth, sourceHeight, value, onChange }: OutpaintCanvasEditorProps) {
  // Cap = +100% per axis, so the max canvas is 3× the source on each side. Fit that into the stage.
  const scale = Math.min(STAGE / (3 * sourceWidth), STAGE / (3 * sourceHeight))
  const stageW = 3 * sourceWidth * scale
  const stageH = 3 * sourceHeight * scale

  const srcLeft = stageW / 2 - (sourceWidth * scale) / 2
  const srcTop = stageH / 2 - (sourceHeight * scale) / 2
  const frameLeft = srcLeft - value.left * scale
  const frameTop = srcTop - value.top * scale
  const frameW = (sourceWidth + value.left + value.right) * scale
  const frameH = (sourceHeight + value.top + value.bottom) * scale

  const outW = toEven(sourceWidth + value.left + value.right)
  const outH = toEven(sourceHeight + value.top + value.bottom)

  const startDrag = useCallback(
    (e: React.PointerEvent, handle: Handle) => {
      e.preventDefault()
      e.stopPropagation()
      const startX = e.clientX
      const startY = e.clientY
      const start = { ...value }
      const move = (ev: PointerEvent) => {
        const dx = (ev.clientX - startX) / scale
        const dy = (ev.clientY - startY) / scale
        const next = { ...start }
        // Dragging the left/top edge outward (negative delta) grows that pad; clamp to [0, source axis].
        if (handle.includes('l')) next.left = clamp(Math.round(start.left - dx), 0, sourceWidth)
        if (handle.includes('r')) next.right = clamp(Math.round(start.right + dx), 0, sourceWidth)
        if (handle.includes('t')) next.top = clamp(Math.round(start.top - dy), 0, sourceHeight)
        if (handle.includes('b')) next.bottom = clamp(Math.round(start.bottom + dy), 0, sourceHeight)
        onChange(next)
      }
      const up = () => {
        window.removeEventListener('pointermove', move)
        window.removeEventListener('pointerup', up)
      }
      window.addEventListener('pointermove', move)
      window.addEventListener('pointerup', up)
    },
    [value, scale, sourceWidth, sourceHeight, onChange],
  )

  const applyPreset = useCallback(
    (ratio: number | null) => {
      if (ratio === null) {
        onChange({ left: 0, right: 0, top: 0, bottom: 0 })
        return
      }
      const current = sourceWidth / sourceHeight
      if (current < ratio) {
        // Too tall for the ratio → widen. Cap each side at +100% (total extra ≤ 2× width).
        const extra = clamp(Math.round(sourceHeight * ratio) - sourceWidth, 0, 2 * sourceWidth)
        const half = Math.round(extra / 2)
        onChange({ left: half, right: extra - half, top: 0, bottom: 0 })
      } else {
        // Too wide → heighten.
        const extra = clamp(Math.round(sourceWidth / ratio) - sourceHeight, 0, 2 * sourceHeight)
        const half = Math.round(extra / 2)
        onChange({ left: 0, right: 0, top: half, bottom: extra - half })
      }
    },
    [sourceWidth, sourceHeight, onChange],
  )

  // A ratio is reachable only if the +100%-per-axis cap can grow the deficient axis far enough.
  const isPresetReachable = (ratio: number | null): boolean => {
    if (ratio === null) return true
    const current = sourceWidth / sourceHeight
    return current < ratio
      ? Math.round(sourceHeight * ratio) - sourceWidth <= 2 * sourceWidth
      : Math.round(sourceWidth / ratio) - sourceHeight <= 2 * sourceHeight
  }

  // Handle display position (centre point) on the frame border, in stage coordinates.
  const handlePos = (h: Handle): { x: number; y: number } => {
    const midX = frameLeft + frameW / 2
    const midY = frameTop + frameH / 2
    const left = frameLeft
    const right = frameLeft + frameW
    const top = frameTop
    const bottom = frameTop + frameH
    switch (h) {
      case 'tl': return { x: left, y: top }
      case 't': return { x: midX, y: top }
      case 'tr': return { x: right, y: top }
      case 'l': return { x: left, y: midY }
      case 'r': return { x: right, y: midY }
      case 'bl': return { x: left, y: bottom }
      case 'b': return { x: midX, y: bottom }
      case 'br': return { x: right, y: bottom }
    }
  }
  const handleCursor: Record<Handle, string> = {
    tl: 'nwse-resize', t: 'ns-resize', tr: 'nesw-resize', l: 'ew-resize',
    r: 'ew-resize', bl: 'nesw-resize', b: 'ns-resize', br: 'nwse-resize',
  }

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-black">
      <div className="flex-1 flex items-center justify-center min-h-0 p-3">
        <div className="relative" style={{ width: stageW, height: stageH }}>
          {/* Output frame: the green-filled generate region around the source. */}
          <div
            className="absolute border border-emerald-500/60 bg-emerald-500/10"
            style={{ left: frameLeft, top: frameTop, width: frameW, height: frameH }}
          />
          {/* Source silhouette (kept region) — fixed size, centred. */}
          <div
            className="absolute bg-zinc-400/80 border border-zinc-300"
            style={{ left: srcLeft, top: srcTop, width: sourceWidth * scale, height: sourceHeight * scale }}
          />
          {HANDLES.map((h) => {
            const { x, y } = handlePos(h)
            return (
              <div
                key={h}
                onPointerDown={(e) => startDrag(e, h)}
                className="absolute w-2.5 h-2.5 -ml-1.5 -mt-1.5 rounded-sm bg-white border border-zinc-500 shadow"
                style={{ left: x, top: y, cursor: handleCursor[h], touchAction: 'none' }}
              />
            )
          })}
        </div>
      </div>
      <div className="px-3 py-2 border-t border-zinc-800 flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-1">
          {PRESETS.map((p) => {
            // Unreachable presets stay clickable but cap-to-closest (applyPreset clamps to the
            // +100% extend cap) — marked with a · and a tooltip so it's not a silent wrong ratio.
            const capped = !isPresetReachable(p.ratio)
            return (
              <button
                key={p.id}
                onClick={() => applyPreset(p.ratio)}
                title={capped ? 'Limited by the +100% expand cap — lands at the closest ratio' : undefined}
                className="px-2 py-0.5 rounded text-[10px] text-zinc-400 hover:text-white hover:bg-zinc-800 border border-zinc-700 transition-colors"
              >
                {p.label}
                {capped && <span className="text-amber-500/80">·</span>}
              </button>
            )
          })}
        </div>
        <span className="text-[10px] text-zinc-500 tabular-nums">
          {sourceWidth}×{sourceHeight} → {outW}×{outH}
        </span>
      </div>
    </div>
  )
}
