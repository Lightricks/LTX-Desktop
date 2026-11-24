/**
 * Motion Track path: the point operations of the editor.
 *
 * The sampler and the types are in motionPathSampler.ts. Nothing here changes the frames the
 * backend draws for a given path. Handles are relative to their anchor.
 */

import {
  isZeroHandle,
  MOTION_PATH_MAX_POINTS,
  type MotionPathPoint,
  type MotionPathVec2,
} from "./motionPathSampler.ts";

/** Round-handle length as a fraction of the neighbouring segment. A smooth look with no overshoot. */
export const MOTION_PATH_ROUND_TENSION = 1 / 3;

export function normalizeVec(x: number, y: number): MotionPathVec2 | null {
  const len = Math.hypot(x, y);
  return len < 1e-8 ? null : { x: x / len, y: y / len };
}

function areCollinear(a: MotionPathVec2, b: MotionPathVec2): boolean {
  return Math.abs(a.x * b.y - a.y * b.x) < 1e-6;
}

function withHandles(
  point: MotionPathPoint,
  handles: { handleIn?: MotionPathVec2; handleOut?: MotionPathVec2 },
): MotionPathPoint {
  const { handleIn: _in, handleOut: _out, ...rest } = point;
  return {
    ...rest,
    ...(handles.handleIn && !isZeroHandle(handles.handleIn) ? { handleIn: handles.handleIn } : {}),
    ...(handles.handleOut && !isZeroHandle(handles.handleOut)
      ? { handleOut: handles.handleOut }
      : {}),
  };
}

/** Remove a point by id. The new last point loses its outgoing handle. */
export function deleteMotionPathPoint(points: MotionPathPoint[], pointId: string): MotionPathPoint[] {
  if (!points.some((p) => p.id === pointId)) return points;
  const next = points.filter((p) => p.id !== pointId);
  const last = next[next.length - 1];
  if (last?.handleOut !== undefined) {
    next[next.length - 1] = withHandles(last, { handleIn: last.handleIn });
  }
  return next;
}

/** Append a point. The new segment is a corner until the user bends it. */
export function addMotionPathPoint(
  points: MotionPathPoint[],
  point: Omit<MotionPathPoint, "handleIn" | "handleOut">,
): MotionPathPoint[] {
  if (points.length >= MOTION_PATH_MAX_POINTS) {
    throw new Error(`Cannot add more than ${MOTION_PATH_MAX_POINTS} points`);
  }
  return [...points, { ...point }];
}

/** Set relative handles on a point. `null` clears that side. */
export function setMotionPathPointHandles(
  points: MotionPathPoint[],
  pointIndex: number,
  handles: { handleIn?: MotionPathVec2 | null; handleOut?: MotionPathVec2 | null },
): MotionPathPoint[] {
  if (pointIndex < 0 || pointIndex >= points.length) return points;
  return points.map((point, index) => {
    if (index !== pointIndex) return point;
    return withHandles(point, {
      handleIn: handles.handleIn === null ? undefined : (handles.handleIn ?? point.handleIn),
      handleOut: handles.handleOut === null ? undefined : (handles.handleOut ?? point.handleOut),
    });
  });
}

/** True when the point is smooth: both handles present and collinear. */
function isMotionPathPointSmooth(points: MotionPathPoint[], index: number): boolean {
  if (index < 0 || index >= points.length || points.length < 2) return false;
  const point = points[index];
  if (index === 0) return !isZeroHandle(point.handleOut);
  if (index === points.length - 1) return !isZeroHandle(point.handleIn);
  if (!point.handleIn || !point.handleOut) return false;
  if (isZeroHandle(point.handleIn) || isZeroHandle(point.handleOut)) return false;
  return areCollinear(point.handleIn, point.handleOut);
}

/** Move an anchor. The handles are relative, so they stay attached. */
export function moveMotionPathPoint(
  points: MotionPathPoint[],
  pointId: string,
  x: number,
  y: number,
): MotionPathPoint[] {
  return points.map((point) => (point.id === pointId ? { ...point, x, y } : point));
}

export type MotionPathHandleSide = "in" | "out";

/** Drag a handle tip (absolute coordinates). A smooth point keeps the other handle mirrored. */
export function moveMotionPathHandle(
  points: MotionPathPoint[],
  params: { pointIndex: number; side: MotionPathHandleSide; x: number; y: number },
): MotionPathPoint[] {
  const { pointIndex, side, x, y } = params;
  if (pointIndex < 0 || pointIndex >= points.length) return points;
  const point = points[pointIndex];
  const relative = { x: x - point.x, y: y - point.y };
  const isEnd = pointIndex === 0 || pointIndex === points.length - 1;
  if (!isMotionPathPointSmooth(points, pointIndex) || isEnd) {
    return setMotionPathPointHandles(points, pointIndex, {
      handleIn: side === "in" ? relative : point.handleIn,
      handleOut: side === "out" ? relative : point.handleOut,
    });
  }
  const dir = normalizeVec(relative.x, relative.y);
  if (!dir) {
    return setMotionPathPointHandles(points, pointIndex, {
      handleIn: side === "in" ? relative : null,
      handleOut: side === "out" ? relative : null,
    });
  }
  const len = Math.hypot(relative.x, relative.y);
  const mirrored = { x: -dir.x * len, y: -dir.y * len };
  return setMotionPathPointHandles(points, pointIndex, {
    handleIn: side === "in" ? relative : mirrored,
    handleOut: side === "out" ? relative : mirrored,
  });
}

/**
 * Make a point round: a symmetric handle pair along the chord of its neighbours. An end point
 * gets one handle toward its neighbour.
 */
export function roundMotionPathPoint(points: MotionPathPoint[], index: number): MotionPathPoint[] {
  if (index < 0 || index >= points.length || points.length < 2) return points;
  const curr = points[index];
  const prev = index > 0 ? points[index - 1] : curr;
  const next = index < points.length - 1 ? points[index + 1] : curr;
  const d1 = Math.hypot(prev.x - curr.x, prev.y - curr.y);
  const d2 = Math.hypot(next.x - curr.x, next.y - curr.y);
  const tension = MOTION_PATH_ROUND_TENSION;
  if (index === 0) {
    if (d2 < 1e-8) return points;
    return setMotionPathPointHandles(points, index, {
      handleOut: { x: (next.x - curr.x) * tension, y: (next.y - curr.y) * tension },
    });
  }
  if (index === points.length - 1) {
    if (d1 < 1e-8) return points;
    return setMotionPathPointHandles(points, index, {
      handleIn: { x: (prev.x - curr.x) * tension, y: (prev.y - curr.y) * tension },
    });
  }
  const tangent = normalizeVec(next.x - prev.x, next.y - prev.y);
  if (!tangent) return points;
  const handleLen = ((d1 + d2) / 2) * tension;
  return setMotionPathPointHandles(points, index, {
    handleIn: { x: -tangent.x * handleLen, y: -tangent.y * handleLen },
    handleOut: { x: tangent.x * handleLen, y: tangent.y * handleLen },
  });
}

/** Toggle sharp and round. */
export function toggleMotionPathPointSmooth(
  points: MotionPathPoint[],
  index: number,
): MotionPathPoint[] {
  if (index < 0 || index >= points.length || points.length < 2) return points;
  if (isMotionPathPointSmooth(points, index)) {
    return setMotionPathPointHandles(points, index, { handleIn: null, handleOut: null });
  }
  return roundMotionPathPoint(points, index);
}

/** Bend a segment so it passes near `point` at parameter `t`. Both side handles aim at it. */
export function setSegmentBend(
  points: MotionPathPoint[],
  segmentIndex: number,
  point: MotionPathVec2,
  t: number,
): MotionPathPoint[] {
  if (segmentIndex < 0 || segmentIndex >= points.length - 1) return points;
  const a = points[segmentIndex];
  const b = points[segmentIndex + 1];
  const clampedT = Math.min(0.85, Math.max(0.15, t));
  const strength = 2 / 3;
  const withOut = setMotionPathPointHandles(points, segmentIndex, {
    handleOut: {
      x: (point.x - a.x) * strength * (1 - clampedT + 0.5),
      y: (point.y - a.y) * strength * (1 - clampedT + 0.5),
    },
  });
  return setMotionPathPointHandles(withOut, segmentIndex + 1, {
    handleIn: {
      x: (point.x - b.x) * strength * (clampedT + 0.5),
      y: (point.y - b.y) * strength * (clampedT + 0.5),
    },
  });
}

/** Clear the bend of a segment: A.handleOut and B.handleIn. */
export function clearSegmentBend(points: MotionPathPoint[], segmentIndex: number): MotionPathPoint[] {
  if (segmentIndex < 0 || segmentIndex >= points.length - 1) return points;
  const withoutOut = setMotionPathPointHandles(points, segmentIndex, { handleOut: null });
  return setMotionPathPointHandles(withoutOut, segmentIndex + 1, { handleIn: null });
}
