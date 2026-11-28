import {
  type MotionPathPoint,
  absoluteHandle,
  closestPointOnSegment,
} from "../../../lib/motionPathSampler.ts";
import { type MotionPathHandleSide } from "../../../lib/motionPathEdit.ts";

import {
  EDITOR_DOT_RADIUS_PX,
  EDITOR_HANDLE_RADIUS_PX,
} from "./motionPathEditorConstants.ts";

const SEGMENT_HIT_THRESHOLD_PX = 14;
/** Hit target slightly larger than the drawn radius for easier grabbing. */
const POINT_HIT_PADDING_PX = 6;
const HANDLE_HIT_PADDING_PX = 6;

function distanceSq(ax: number, ay: number, bx: number, by: number): number {
  const dx = ax - bx;
  const dy = ay - by;
  return dx * dx + dy * dy;
}

export function hitTestPointIndex(
  points: MotionPathPoint[],
  px: number,
  py: number,
  width: number,
  height: number,
): number {
  const radiusSq = (EDITOR_DOT_RADIUS_PX + POINT_HIT_PADDING_PX) ** 2;
  return points.findIndex((point) => {
    const pointPx = point.x * width;
    const pointPy = point.y * height;
    return distanceSq(px, py, pointPx, pointPy) <= radiusSq;
  });
}

export type HandleHit = {
  pointIndex: number;
  side: MotionPathHandleSide;
};

/** Hit-test handleIn / handleOut tips that belong to the selected point only. */
export function hitTestSelectedPointHandle(
  points: MotionPathPoint[],
  selectedIndex: number,
  px: number,
  py: number,
  width: number,
  height: number,
): HandleHit | null {
  if (selectedIndex < 0 || selectedIndex >= points.length) return null;
  const radiusSq = (EDITOR_HANDLE_RADIUS_PX + HANDLE_HIT_PADDING_PX) ** 2;
  const point = points[selectedIndex];

  const outTip = absoluteHandle(point, point.handleOut);
  if (outTip && distanceSq(px, py, outTip.x * width, outTip.y * height) <= radiusSq) {
    return { pointIndex: selectedIndex, side: "out" };
  }

  const inTip = absoluteHandle(point, point.handleIn);
  if (inTip && distanceSq(px, py, inTip.x * width, inTip.y * height) <= radiusSq) {
    return { pointIndex: selectedIndex, side: "in" };
  }

  return null;
}

export function hitTestSegment(
  points: MotionPathPoint[],
  px: number,
  py: number,
  width: number,
  height: number,
): { segmentIndex: number; t: number } | null {
  if (points.length < 2 || width <= 0 || height <= 0) return null;
  const nx = px / width;
  const ny = py / height;
  const thresholdSq = (SEGMENT_HIT_THRESHOLD_PX / Math.min(width, height)) ** 2;

  let best: { segmentIndex: number; t: number; distSq: number } | null = null;
  for (let i = 0; i < points.length - 1; i++) {
    const hit = closestPointOnSegment(points[i], points[i + 1], nx, ny);
    if (hit.distSq > thresholdSq) continue;
    if (!best || hit.distSq < best.distSq) {
      best = { segmentIndex: i, t: hit.t, distSq: hit.distSq };
    }
  }
  return best ? { segmentIndex: best.segmentIndex, t: best.t } : null;
}
