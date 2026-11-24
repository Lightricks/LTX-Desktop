import {
  findNearestFreeFrameIndex,
  parseKeyframeSeconds,
  parseKeyframeTimecode,
} from './keyframe-timeline.ts'

type PositionedKeyframe = {
  id: string
  frameIndex: number
}

function placementContext(
  keyframes: readonly PositionedKeyframe[],
  id: string,
): { current: PositionedKeyframe; others: PositionedKeyframe[] } | null {
  const current = keyframes.find((keyframe) => keyframe.id === id)
  if (!current) return null

  return {
    current,
    others: keyframes.filter((keyframe) => keyframe.id !== id),
  }
}

export function nudgeKeyframe(
  keyframes: readonly PositionedKeyframe[],
  id: string,
  delta: -1 | 1,
  lastFrame: number,
): number | null {
  const context = placementContext(keyframes, id)
  if (!context) return null

  return findNearestFreeFrameIndex(
    context.others,
    context.current.frameIndex + delta,
    lastFrame,
  )
}

function placeNearest(
  keyframes: readonly PositionedKeyframe[],
  id: string,
  wantedFrame: number | null,
  lastFrame: number,
): number | null {
  const context = placementContext(keyframes, id)
  if (!context || wantedFrame === null) return null

  return findNearestFreeFrameIndex(context.others, wantedFrame, lastFrame)
}

/** The frame a typed `MM:SS.FF` time gives the keyframe, moved to the nearest free frame. */
export function applyTimecode(
  keyframes: readonly PositionedKeyframe[],
  id: string,
  value: string,
  fps: number,
  lastFrame: number,
): number | null {
  return placeNearest(keyframes, id, parseKeyframeTimecode(value, fps), lastFrame)
}

/** The frame typed seconds give the keyframe, moved to the nearest free frame inside the clip. */
export function applySeconds(
  keyframes: readonly PositionedKeyframe[],
  id: string,
  value: string,
  fps: number,
  lastFrame: number,
): number | null {
  return placeNearest(keyframes, id, parseKeyframeSeconds(value, fps), lastFrame)
}
