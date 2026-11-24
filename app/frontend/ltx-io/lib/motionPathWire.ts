/**
 * The two shapes of a Motion Track path. The wire path is what the create body and the saved
 * spec hold. The editor path adds an `id` to each point. These functions are the only place
 * that convert between them.
 */

import { randomUUID } from "../../lib/random-uuid.ts";
import {
  MOTION_PATH_EASING_OPTIONS,
  MOTION_PATH_MAX_HOLD_SEC,
  MOTION_PATH_MAX_POINTS,
  type MotionPath,
  type MotionPathVec2,
  type WireMotionPath,
} from "./motionPathSampler.ts";

export function toEditorPath(path: WireMotionPath): MotionPath {
  return {
    ...path,
    points: path.points.map(({ handleIn, handleOut, ...anchor }) => ({
      id: randomUUID(),
      ...anchor,
      ...(handleIn ? { handleIn } : {}),
      ...(handleOut ? { handleOut } : {}),
    })),
  };
}

export function toWireMotionPath(path: MotionPath): WireMotionPath {
  return { ...path, points: path.points.map(({ id: _id, ...point }) => point) };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readVec(value: unknown): MotionPathVec2 | null {
  return isRecord(value) && typeof value.x === "number" && typeof value.y === "number"
    ? { x: value.x, y: value.y }
    : null;
}

function readHoldSec(value: unknown): number | null {
  return typeof value === "number" && value >= 0 && value <= MOTION_PATH_MAX_HOLD_SEC ? value : null;
}

/** The path of a saved spec, or null when it is not a path the backend accepts. */
export function readWireMotionPath(value: unknown): WireMotionPath | null {
  if (!isRecord(value) || !Array.isArray(value.points)) return null;
  const easing = MOTION_PATH_EASING_OPTIONS.find((option) => option === value.easing);
  const startHoldSec = readHoldSec(value.startHoldSec);
  const endHoldSec = readHoldSec(value.endHoldSec);
  if (easing == null || startHoldSec == null || endHoldSec == null) return null;
  if (value.points.length < 2 || value.points.length > MOTION_PATH_MAX_POINTS) return null;
  const points: WireMotionPath["points"] = [];
  for (const raw of value.points) {
    const anchor = readVec(raw);
    if (anchor == null || !isRecord(raw)) return null;
    const handleIn = readVec(raw.handleIn);
    const handleOut = readVec(raw.handleOut);
    points.push({
      ...anchor,
      ...(handleIn ? { handleIn } : {}),
      ...(handleOut ? { handleOut } : {}),
    });
  }
  return { points, easing, startHoldSec, endHoldSec };
}
