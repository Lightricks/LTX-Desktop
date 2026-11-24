/**
 * Motion Track path: the types, the pace, and the frames of the moving dot.
 *
 * Port of the ltx.io sampler. The backend twin is
 * backend/services/features/video/motion_path.py. shared/motion-path-golden.json pins both
 * to the same frames. The editor-only point operations are in motionPathEdit.ts.
 *
 * Coordinates are normalized 0-1 over the frame. Each point owns relative handles, as in
 * Paper.js. The segment A to B is a cubic Bézier from A + A.handleOut to B + B.handleIn.
 * A point is smooth when both handles are set and collinear.
 */

import motionPathShared from "../../../shared/motion-path.json" with { type: "json" };

import type { components } from "../../generated/backend-openapi";

type WireSchemas = components["schemas"];

export type MotionPathVec2 = WireSchemas["MotionPathVec2"];
export type MotionPathEasing = WireSchemas["MotionPath"]["easing"];
/** The path as the create body and the saved spec hold it. */
export type WireMotionPath = WireSchemas["MotionPath"];

/** An editor point. The `id` stays in the editor: `toWireMotionPath` drops it. */
export type MotionPathPoint = {
  id: string;
  x: number;
  y: number;
  /** Relative incoming handle. Omitted or zero means no in-tangent. */
  handleIn?: MotionPathVec2;
  /** Relative outgoing handle. Omitted or zero means no out-tangent. */
  handleOut?: MotionPathVec2;
};

/** The wire path with an `id` on each point. */
export type MotionPath = Omit<WireMotionPath, "points"> & { points: MotionPathPoint[] };

export const MOTION_PATH_MAX_POINTS = 7;
/** The most paths in one run (`MOTION_PATH_MAX_PATHS` in the backend). */
export const MOTION_PATH_MAX_TRACKS = 3;
/** The preview plays at this rate. The clip rate does not change the motion. */
export const MOTION_PATH_PREVIEW_FPS = 24;

/** Overall timing curve along the path. */
export const MOTION_PATH_EASING_OPTIONS = [
  "linear",
  "easeIn",
  "easeOut",
  "easeInOut",
  "easeInBack",
  "easeOutBack",
  "easeInOutBack",
  "gentle",
  "quick",
  "bouncy",
  "slow",
] as const satisfies readonly MotionPathEasing[];
export const MOTION_PATH_DEFAULT_EASING: MotionPathEasing = "easeInOut";

/**
 * The pauses the editor offers, in seconds. The backend takes any pause from 0 to
 * `MOTION_PATH_MAX_HOLD_SEC`, so a saved path can hold another value.
 */
export const MOTION_PATH_HOLD_OPTIONS_SEC = [0, 0.5, 1, 1.5, 2, 3] as const;
export const MOTION_PATH_MAX_HOLD_SEC = 3;

/**
 * The dot color of each track in the control clip and in the editor. The model sees only the
 * clip, so the editor draws these colors and not a theme token. The backend reads the same file.
 */
export const MOTION_PATH_TRACK_DOT_COLORS: readonly string[] = motionPathShared.dotColors;

/** The dot color of a track. `trackIndex` counts the tracks that have a path, as the clip does. */
export function motionPathDotColor(trackIndex: number): string {
  return MOTION_PATH_TRACK_DOT_COLORS[trackIndex % MOTION_PATH_TRACK_DOT_COLORS.length];
}

const EASING_C1 = 1.70158;
const EASING_C2 = EASING_C1 * 1.525;
const EASING_C3 = EASING_C1 + 1;

export function clamp01(n: number): number {
  return Math.min(1, Math.max(0, n));
}

export function isZeroHandle(handle: MotionPathVec2 | undefined): boolean {
  return !handle || (Math.abs(handle.x) < 1e-8 && Math.abs(handle.y) < 1e-8);
}

/** Absolute position of a relative handle tip, or null for no handle. */
export function absoluteHandle(
  point: MotionPathVec2,
  handle: MotionPathVec2 | undefined,
): MotionPathVec2 | null {
  return !handle || isZeroHandle(handle) ? null : { x: point.x + handle.x, y: point.y + handle.y };
}

/** easings.net `easeOutBounce`. */
function easeOutBounce(x: number): number {
  const n1 = 7.5625;
  const d1 = 2.75;
  if (x < 1 / d1) return n1 * x * x;
  if (x < 2 / d1) return n1 * (x -= 1.5 / d1) * x + 0.75;
  if (x < 2.5 / d1) return n1 * (x -= 2.25 / d1) * x + 0.9375;
  return n1 * (x -= 2.625 / d1) * x + 0.984375;
}

/** Map progress 0-1 through a named easing curve. Back and bounce leave [0, 1]. */
export function applyMotionPathEasing(t: number, easing: MotionPathEasing): number {
  const x = clamp01(t);
  switch (easing) {
    case "easeIn":
      return x * x * x;
    case "easeOut":
      return 1 - Math.pow(1 - x, 3);
    case "easeInOut":
      return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
    case "easeInBack":
      return EASING_C3 * x * x * x - EASING_C1 * x * x;
    case "easeOutBack":
      return 1 + EASING_C3 * Math.pow(x - 1, 3) + EASING_C1 * Math.pow(x - 1, 2);
    case "easeInOutBack":
      return x < 0.5
        ? (Math.pow(2 * x, 2) * ((EASING_C2 + 1) * 2 * x - EASING_C2)) / 2
        : (Math.pow(2 * x - 2, 2) * ((EASING_C2 + 1) * (x * 2 - 2) + EASING_C2) + 2) / 2;
    case "gentle":
      return Math.sin((x * Math.PI) / 2);
    case "quick":
      return x === 1 ? 1 : 1 - Math.pow(2, -10 * x);
    case "bouncy":
      return easeOutBounce(x);
    case "slow":
      return -(Math.cos(Math.PI * x) - 1) / 2;
    case "linear":
      return x;
    default: {
      const unhandled: never = easing;
      throw new Error(`Unknown easing: ${String(unhandled)}`);
    }
  }
}

function snapHoldDown(maxAllowed: number): number {
  return MOTION_PATH_HOLD_OPTIONS_SEC.reduce<number>(
    (best, option) => (option <= maxAllowed ? option : best),
    0,
  );
}

/**
 * Keep both pauses under the duration (`sum < durationSec`), as the backend requires.
 * The start pause stays when it can. The end pause shrinks first.
 */
export function clampMotionPathHolds(params: {
  durationSec: number;
  startHoldSec: number;
  endHoldSec: number;
}): { startHoldSec: number; endHoldSec: number } {
  const { durationSec } = params;
  let { startHoldSec, endHoldSec } = params;
  if (startHoldSec + endHoldSec < durationSec) return { startHoldSec, endHoldSec };
  endHoldSec = snapHoldDown(Math.min(endHoldSec, durationSec - startHoldSec - 1e-9));
  if (startHoldSec + endHoldSec >= durationSec) {
    startHoldSec = snapHoldDown(durationSec - endHoldSec - 1e-9);
  }
  return { startHoldSec, endHoldSec };
}

/** Cubic Bézier: (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3. */
function sampleCubicBezier(
  p0: MotionPathVec2,
  p1: MotionPathVec2,
  p2: MotionPathVec2,
  p3: MotionPathVec2,
  t: number,
): MotionPathVec2 {
  const u = 1 - t;
  return {
    x: u * u * u * p0.x + 3 * u * u * t * p1.x + 3 * u * t * t * p2.x + t * t * t * p3.x,
    y: u * u * u * p0.y + 3 * u * u * t * p1.y + 3 * u * t * t * p2.y + t * t * t * p3.y,
  };
}

function segmentControls(
  a: MotionPathPoint,
  b: MotionPathPoint,
): { c1: MotionPathVec2; c2: MotionPathVec2 } {
  return {
    c1: absoluteHandle(a, a.handleOut) ?? { x: a.x, y: a.y },
    c2: absoluteHandle(b, b.handleIn) ?? { x: b.x, y: b.y },
  };
}

function sampleSegment(points: MotionPathPoint[], segmentIndex: number, t: number): MotionPathVec2 {
  const a = points[segmentIndex];
  const b = points[segmentIndex + 1];
  const { c1, c2 } = segmentControls(a, b);
  return sampleCubicBezier(a, c1, c2, b, t);
}

/** The closest point on a segment, for pointer hit tests. */
export function closestPointOnSegment(
  a: MotionPathPoint,
  b: MotionPathPoint,
  px: number,
  py: number,
  sampleCount = 24,
): { x: number; y: number; t: number; distSq: number } {
  const { c1, c2 } = segmentControls(a, b);
  let best = { x: a.x, y: a.y, t: 0, distSq: Number.POSITIVE_INFINITY };
  for (let i = 0; i <= sampleCount; i++) {
    const t = i / sampleCount;
    const p = sampleCubicBezier(a, c1, c2, b, t);
    const distSq = (p.x - px) ** 2 + (p.y - py) ** 2;
    if (distSq < best.distSq) best = { x: p.x, y: p.y, t, distSq };
  }
  return best;
}

/**
 * The dot at `progress` along the path, each segment taking the same time.
 * Back and bounce easings pass the ends. The dot then follows the end chord, because a
 * cubic does not extrapolate for t outside [0, 1].
 */
function positionAt(points: MotionPathPoint[], progress: number): MotionPathVec2 {
  const segmentCount = points.length - 1;
  if (progress <= 0) {
    const [start, next] = points;
    return {
      x: start.x + (start.x - next.x) * -progress,
      y: start.y + (start.y - next.y) * -progress,
    };
  }
  if (progress >= 1) {
    const end = points[segmentCount];
    const prev = points[segmentCount - 1];
    return {
      x: end.x + (end.x - prev.x) * (progress - 1),
      y: end.y + (end.y - prev.y) * (progress - 1),
    };
  }
  const scaled = progress * segmentCount;
  const segmentIndex = Math.min(segmentCount - 1, Math.floor(scaled));
  return sampleSegment(points, segmentIndex, scaled - segmentIndex);
}

/**
 * The dot of every frame, in frame-normalized coordinates. The caller keeps the pauses under
 * `durationSec` (`clampMotionPathHolds`). A back or bounce easing can leave [0, 1].
 */
export function sampleMotionPathFrames(
  path: MotionPath,
  durationSec: number,
  frameCount: number,
): MotionPathVec2[] {
  const { points, easing, startHoldSec, endHoldSec } = path;
  if (points.length < 2 || points.length > MOTION_PATH_MAX_POINTS) {
    throw new Error(`A path needs 2 to ${MOTION_PATH_MAX_POINTS} points`);
  }
  const lastIndex = frameCount - 1;
  const moveStart = Math.min(lastIndex, Math.round((startHoldSec / durationSec) * lastIndex));
  const moveEnd =
    lastIndex -
    Math.min(lastIndex - moveStart, Math.round((endHoldSec / durationSec) * lastIndex));
  return Array.from({ length: frameCount }, (_, index) => {
    const progress =
      index <= moveStart ? 0 : index >= moveEnd ? 1 : (index - moveStart) / (moveEnd - moveStart);
    return positionAt(points, applyMotionPathEasing(progress, easing));
  });
}
