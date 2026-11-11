/**
 * Frame-grid math for multi-keyframe editing.
 *
 * Positions are integer frame indices in `0..lastFrame`. Two keyframes may
 * never share an index — every placement routes through pickFreeFrameIndex or
 * retimeKeyframes so collisions cannot silently drop images.
 */

type Positioned = { frameIndex: number }

/**
 * Last legal keyframe index for a clip. Must match backend
 * ``frame_math.compute_num_frames(duration, fps) - 1`` — duration×fps rounded
 * to the pipeline's (n-1)%8==0 grid — or 25/50 fps stills can sit past the
 * generated clip and 422.
 */
export function lastFrameFromDuration(durationSeconds: number, fps: number): number {
  const frameCount = Math.max(9, Math.floor((durationSeconds * fps) / 8) * 8 + 1)
  return frameCount - 1
}

export function clampFrameIndex(frameIndex: number, lastFrame: number): number {
  return Math.min(Math.max(0, Math.round(frameIndex)), Math.max(0, lastFrame))
}

/**
 * The keyframe on the playhead, else the last one before it — a gap shows the
 * image still standing rather than going blank. Unordered input is sorted.
 */
export function keyframeAtOrBefore<T extends { frameIndex: number }>(
  keyframes: readonly T[],
  frameIndex: number,
): T | undefined {
  const ordered = [...keyframes].sort((a, b) => a.frameIndex - b.frameIndex)
  let match: T | undefined
  for (const keyframe of ordered) {
    if (keyframe.frameIndex > frameIndex) break
    match = keyframe
  }
  return match ?? ordered[0]
}

export type DraggedFrame = {
  id: string
  frameIndex: number
}

export function sameDraggedFrame(
  left: DraggedFrame | null | undefined,
  right: DraggedFrame | null | undefined,
): boolean {
  return left?.id === right?.id && left?.frameIndex === right?.frameIndex
}

export function withDraggedFrame<T extends { id: string; frameIndex: number }>(
  keyframes: readonly T[],
  drag: DraggedFrame | null,
): readonly T[] {
  if (!drag) return keyframes
  return keyframes.map((keyframe) => (
    keyframe.id === drag.id && keyframe.frameIndex === drag.frameIndex
      ? keyframe
      : keyframe.id === drag.id
        ? { ...keyframe, frameIndex: drag.frameIndex }
        : keyframe
  ))
}

export function previewKeyframeForPlayhead<T extends { id: string; frameIndex: number }>(
  keyframes: readonly T[],
  playheadFrame: number,
  drag: DraggedFrame | null = null,
): T | undefined {
  return keyframeAtOrBefore(withDraggedFrame(keyframes, drag), playheadFrame)
}

/** `MM:SS.FF`, where `FF` is the frame within its second (`0..fps - 1`). */
export function formatKeyframeTimecode(frameIndex: number, fps: number): string {
  const total = Math.max(0, Math.round(frameIndex))
  const frames = total % fps
  const totalSeconds = Math.floor(total / fps)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  const pad = (value: number) => String(value).padStart(2, '0')
  return `${pad(minutes)}:${pad(seconds)}.${pad(frames)}`
}

/**
 * Parses `MM:SS.FF` (frames optional) into a frame index. Returns null for
 * anything unparseable or out of grid.
 */
export function parseKeyframeTimecode(value: string, fps: number): number | null {
  const match = /^(\d{1,2}):(\d{1,2})(?:\.(\d{1,2}))?$/.exec(value.trim())
  if (!match) return null

  const minutes = Number(match[1])
  const seconds = Number(match[2])
  const frames = Number(match[3] ?? '0')
  if (seconds >= 60 || frames >= fps) return null

  return (minutes * 60 + seconds) * fps + frames
}

/** Seconds with two decimals and no minutes, like `1.50s`. A clip is never long enough for minutes. */
export function formatKeyframeSeconds(frameIndex: number, fps: number): string {
  return `${(Math.max(0, Math.round(frameIndex)) / fps).toFixed(2)}s`
}

/**
 * Parses seconds like `1.5`, `1.5s` or `.5` into the nearest frame index. Returns null for
 * anything else. A time past the clip is the caller's to clamp.
 */
export function parseKeyframeSeconds(value: string, fps: number): number | null {
  const match = /^(\d+(?:\.\d*)?|\.\d+)\s*s?$/i.exec(value.trim())
  if (!match) return null
  return Math.round(Number(match[1]) * fps)
}

export type RulerTick = {
  frame: number
  /** Set on a main tick, like `2s`. A small tick between two main ticks has none. */
  label: string | null
}

/** Seconds between two labels, from short to long. The first step that gives few enough labels wins. */
const RULER_LABEL_STEPS = [1, 2, 5, 10, 15, 30, 60]
const RULER_MAX_LABELS = 10

/**
 * The ticks of a ruler for a clip of `lastFrame + 1` frames: a labelled tick at each step and a
 * small tick halfway between. The step grows with the clip so the labels stay apart.
 */
export function rulerTicks(lastFrame: number, fps: number): RulerTick[] {
  // A `NaN` would never end the loop below, so it gets no ticks.
  if (!Number.isFinite(lastFrame) || !Number.isFinite(fps)) return []
  if (lastFrame <= 0 || fps <= 0) return []
  const totalSeconds = lastFrame / fps
  const step =
    RULER_LABEL_STEPS.find((candidate) => totalSeconds / candidate <= RULER_MAX_LABELS) ??
    RULER_LABEL_STEPS[RULER_LABEL_STEPS.length - 1]
  const ticks: RulerTick[] = []
  for (let index = 0; ; index++) {
    const seconds = (index * step) / 2
    const frame = Math.round(seconds * fps)
    if (frame > lastFrame) break
    ticks.push({ frame, label: index % 2 === 0 ? `${seconds}s` : null })
  }
  return ticks
}

export function frameFromPointer(
  clientX: number,
  trackRect: { left: number; width: number },
  lastFrame: number,
): number {
  if (lastFrame <= 0 || trackRect.width <= 0) return 0
  const ratio = (clientX - trackRect.left) / trackRect.width
  return clampFrameIndex(Math.round(ratio * lastFrame), lastFrame)
}

function inRangeTakenFrameSet(keyframes: readonly Positioned[], lastFrame: number): Set<number> {
  return new Set(
    keyframes
      .map(({ frameIndex }) => frameIndex)
      .filter((frameIndex) => frameIndex >= 0 && frameIndex <= lastFrame),
  )
}

/**
 * Where to drop a new keyframe: the preferred frame when free, otherwise the
 * middle of the widest free stretch so burst adds spread across the timeline.
 */
export function pickFreeFrameIndex(
  keyframes: readonly Positioned[],
  lastFrame: number,
  preferredFrame: number,
): number | null {
  const durationFrames = lastFrame + 1
  if (durationFrames <= 0) return null

  const takenSet = inRangeTakenFrameSet(keyframes, lastFrame)
  if (takenSet.size >= durationFrames) return null

  const clampedPreferred = clampFrameIndex(preferredFrame, lastFrame)
  if (!takenSet.has(clampedPreferred)) return clampedPreferred

  let widest: { start: number; length: number } | undefined
  let runStart: number | undefined
  for (let frame = 0; frame <= durationFrames; frame++) {
    const isFree = frame < durationFrames && !takenSet.has(frame)
    if (isFree) {
      runStart ??= frame
      continue
    }
    if (runStart === undefined) continue
    const length = frame - runStart
    if (!widest || length > widest.length) widest = { start: runStart, length }
    runStart = undefined
  }

  if (!widest) return null
  return widest.start + Math.floor((widest.length - 1) / 2)
}

/**
 * Where a dragged or retyped keyframe lands: the target frame, or the closest
 * free one either side. Never jumps to an unrelated gap on the timeline.
 */
export function findNearestFreeFrameIndex(
  keyframes: readonly Positioned[],
  targetFrame: number,
  lastFrame: number,
): number | null {
  const durationFrames = lastFrame + 1
  if (durationFrames <= 0) return null

  const takenSet = inRangeTakenFrameSet(keyframes, lastFrame)
  const target = clampFrameIndex(targetFrame, lastFrame)
  for (let offset = 0; offset < durationFrames; offset++) {
    const before = target - offset
    const after = target + offset
    if (after <= lastFrame && !takenSet.has(after)) return after
    if (before >= 0 && !takenSet.has(before)) return before
  }
  return null
}

/**
 * Re-grids keyframes after a frame-rate or duration change. Scales each
 * keyframe by oldLastFrame → newLastFrame so a marker at 50% of a 5s clip
 * stays at 50% of a 10s clip (and the reverse). Collisions still spread to
 * unique frames; extras past the slot count are dropped.
 */
export function retimeKeyframes<T extends { frameIndex: number }>(
  keyframes: readonly T[],
  oldLastFrame: number,
  newLastFrame: number,
): T[] {
  const scale = oldLastFrame <= 0 ? 0 : newLastFrame / oldLastFrame

  const targets = keyframes
    .map((keyframe, order) => ({
      keyframe,
      order,
      target: clampFrameIndex(keyframe.frameIndex * scale, newLastFrame),
    }))
    .sort((a, b) => a.target - b.target || a.order - b.order)

  const positions = targets.map(({ target }) => target)
  for (let i = 1; i < positions.length; i++) {
    positions[i] = Math.max(positions[i], positions[i - 1] + 1)
  }
  let ceiling = newLastFrame
  for (let i = positions.length - 1; i >= 0; i--) {
    positions[i] = Math.max(0, Math.min(positions[i], ceiling))
    ceiling = positions[i] - 1
  }

  const used = new Set<number>()
  const retimed: T[] = []
  for (let i = 0; i < targets.length; i++) {
    const frameIndex = positions[i]
    if (used.has(frameIndex)) continue
    used.add(frameIndex)
    retimed.push({ ...targets[i].keyframe, frameIndex })
  }
  return retimed
}

type TimelineSettings = {
  duration: number | null
  fps: number
}

export function retimeKeyframesForSettings<T extends { frameIndex: number }>(
  keyframes: readonly T[],
  previous: TimelineSettings,
  next: TimelineSettings,
): T[] {
  const fallbackDuration = previous.duration ?? next.duration
  const fpsRatio = next.fps / previous.fps
  const oldLastFrame = fallbackDuration === null
    ? Math.max(0, ...keyframes.map(({ frameIndex }) => frameIndex))
    : lastFrameFromDuration(previous.duration ?? fallbackDuration, previous.fps)
  const newLastFrame = fallbackDuration === null
    ? Math.max(0, ...keyframes.map(({ frameIndex }) => Math.round(frameIndex * fpsRatio)))
    : lastFrameFromDuration(next.duration ?? fallbackDuration, next.fps)

  return retimeKeyframes(keyframes, oldLastFrame, newLastFrame)
}
