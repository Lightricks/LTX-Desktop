import { roundMotionPathPoint } from "../../../lib/motionPathEdit.ts";
import {
  clamp01,
  MOTION_PATH_MAX_POINTS,
  type MotionPathPoint,
  type MotionPathVec2,
} from "../../../lib/motionPathSampler.ts";
import { randomUUID } from "../../../../lib/random-uuid.ts";

/** Cap raw stroke size before RDP to keep simplification fast. */
const FREEHAND_RDP_MAX_INPUT_POINTS = 400;

function distance(a: MotionPathVec2, b: MotionPathVec2): number {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

/** Perpendicular distance from point `p` to the segment `a`→`b`. */
function perpendicularDistance(
  p: MotionPathVec2,
  a: MotionPathVec2,
  b: MotionPathVec2,
): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lenSq = dx * dx + dy * dy;
  if (lenSq < 1e-16) return distance(p, a);
  const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / lenSq));
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

/**
 * One light moving-average pass — enough to calm tablet jitter without
 * washing out the shape of the stroke.
 */
function lightSmoothStroke(stroke: MotionPathVec2[]): MotionPathVec2[] {
  if (stroke.length < 3) return stroke.map((point) => ({ ...point }));
  const next: MotionPathVec2[] = [{ ...stroke[0] }];
  for (let i = 1; i < stroke.length - 1; i++) {
    next.push({
      x: (stroke[i - 1].x + stroke[i].x * 2 + stroke[i + 1].x) / 4,
      y: (stroke[i - 1].y + stroke[i].y * 2 + stroke[i + 1].y) / 4,
    });
  }
  next.push({ ...stroke[stroke.length - 1] });
  return next;
}

/**
 * Ramer–Douglas–Peucker: keep the points that define the stroke’s shape
 * (corners / bends), drop the ones that lie nearly on a straight span.
 */
function simplifyDouglasPeucker(
  stroke: MotionPathVec2[],
  epsilon: number,
): MotionPathVec2[] {
  if (stroke.length <= 2) return stroke.map((point) => ({ ...point }));

  let maxDist = 0;
  let maxIndex = 0;
  const first = stroke[0];
  const last = stroke[stroke.length - 1];
  for (let i = 1; i < stroke.length - 1; i++) {
    const dist = perpendicularDistance(stroke[i], first, last);
    if (dist > maxDist) {
      maxDist = dist;
      maxIndex = i;
    }
  }

  if (maxDist <= epsilon) {
    return [{ ...first }, { ...last }];
  }

  const left = simplifyDouglasPeucker(stroke.slice(0, maxIndex + 1), epsilon);
  const right = simplifyDouglasPeucker(stroke.slice(maxIndex), epsilon);
  return [...left.slice(0, -1), ...right];
}

/**
 * Binary-search an RDP epsilon so we land at most `maxPoints` anchors while
 * keeping as much of the user’s shape as possible.
 */
function simplifyToMaxPoints(
  stroke: MotionPathVec2[],
  maxPoints: number,
): MotionPathVec2[] {
  if (stroke.length <= maxPoints) {
    return stroke.map((point) => ({ ...point }));
  }

  let minX = stroke[0].x;
  let maxX = stroke[0].x;
  let minY = stroke[0].y;
  let maxY = stroke[0].y;
  for (const point of stroke) {
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  }
  const extent = Math.hypot(maxX - minX, maxY - minY);
  if (extent < 1e-8) {
    return [{ ...stroke[0] }, { ...stroke[stroke.length - 1] }].slice(0, maxPoints);
  }

  let low = 0;
  let high = extent;
  let best = simplifyDouglasPeucker(stroke, high);
  for (let i = 0; i < 18; i++) {
    const mid = (low + high) / 2;
    const candidate = simplifyDouglasPeucker(stroke, mid);
    if (candidate.length > maxPoints) {
      low = mid;
    } else {
      high = mid;
      best = candidate;
    }
  }

  // Safety: if still over (degenerate cases), thin by even arc-length pick.
  if (best.length > maxPoints) {
    return resamplePolylineByArcLength(best, maxPoints);
  }
  return best;
}

/** Evenly resample a polyline by arc length into `count` points (inclusive ends). */
function resamplePolylineByArcLength(
  stroke: MotionPathVec2[],
  count: number,
): MotionPathVec2[] {
  if (stroke.length === 0 || count < 1) return [];
  if (stroke.length === 1 || count === 1) return [{ ...stroke[0] }];

  const segments: number[] = [0];
  let total = 0;
  for (let i = 1; i < stroke.length; i++) {
    total += distance(stroke[i - 1], stroke[i]);
    segments.push(total);
  }
  if (total < 1e-8) {
    return Array.from({ length: count }, () => ({ ...stroke[0] }));
  }

  const result: MotionPathVec2[] = [];
  let seg = 1;
  for (let i = 0; i < count; i++) {
    const target = (total * i) / (count - 1);
    while (seg < segments.length && segments[seg] < target) seg++;
    const a = stroke[seg - 1];
    const b = stroke[Math.min(seg, stroke.length - 1)];
    const segStart = segments[seg - 1];
    const segEnd = segments[Math.min(seg, segments.length - 1)];
    const span = segEnd - segStart;
    const t = span < 1e-8 ? 0 : (target - segStart) / span;
    result.push({
      x: a.x + (b.x - a.x) * t,
      y: a.y + (b.y - a.y) * t,
    });
  }
  return result;
}

/**
 * Convert a freehand stroke (normalized 0–1) into simplified, smooth Motion
 * Path points that stay close to what the user drew.
 */
function freehandStrokeToMotionPathPoints(stroke: MotionPathVec2[]): MotionPathPoint[] {
  if (stroke.length === 0) return [];

  let calmed = lightSmoothStroke(stroke);
  if (calmed.length > FREEHAND_RDP_MAX_INPUT_POINTS) {
    calmed = resamplePolylineByArcLength(calmed, FREEHAND_RDP_MAX_INPUT_POINTS);
  }
  const simplified = simplifyToMaxPoints(calmed, MOTION_PATH_MAX_POINTS);

  // Prefer at least 2 points; if the stroke was tiny, keep the ends.
  const sampled =
    simplified.length >= 2
      ? simplified
      : resamplePolylineByArcLength(calmed, 2);

  const points: MotionPathPoint[] = sampled.map((sample) => ({
    id: randomUUID(),
    x: clamp01(sample.x),
    y: clamp01(sample.y),
  }));

  // Freehand always makes round joins: a curved line, not a sharp polyline.
  return points.reduce((next, _point, index) => roundMotionPathPoint(next, index), points);
}

/** Free draw always replaces the current path with the new simplified stroke. */
export function commitFreehandStroke(stroke: MotionPathVec2[]): MotionPathPoint[] | null {
  if (stroke.length < 2) return null;
  const next = freehandStrokeToMotionPathPoints(stroke);
  return next.length >= 2 ? next : null;
}
