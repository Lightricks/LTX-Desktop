import {
  addMotionPathPoint, clearSegmentBend, deleteMotionPathPoint, moveMotionPathHandle,
  moveMotionPathPoint, setMotionPathPointHandles, setSegmentBend, toggleMotionPathPointSmooth,
} from "../../../lib/motionPathEdit.ts";
import {
  clamp01, MOTION_PATH_MAX_POINTS, type MotionPathPoint, type MotionPathVec2,
} from "../../../lib/motionPathSampler.ts";
import { randomUUID } from "../../../../lib/random-uuid.ts";
import { commitFreehandStroke } from "./freehandToMotionPath.ts";
import {
  type HandleHit, hitTestPointIndex, hitTestSegment, hitTestSelectedPointHandle,
} from "./motionPathHitTest.ts";

export type MotionPathToolMode = "pen" | "erase" | "convert" | "freehand";
export type CanvasLocalPointer = { px: number; py: number; width: number; height: number };

export type MotionPathGesture =
  | { kind: "idle" }
  | { kind: "freehand"; stroke: MotionPathVec2[] }
  | { kind: "place"; pointIndex: number; originPx: number; originPy: number; curved: boolean }
  | { kind: "dragHandle"; hit: HandleHit; hasHistory: boolean }
  | { kind: "dragPoint"; pointId: string; hasHistory: boolean }
  | { kind: "dragCurve"; segmentIndex: number; t: number };

export type MotionPathPointsEdit = {
  points: MotionPathPoint[];
  historyMode: "push" | "replace";
};

export type PointerMoveOutcome = {
  gesture: MotionPathGesture;
  pointsEdit?: MotionPathPointsEdit;
  activeSegment?: number | null;
  activeHandle?: HandleHit | null;
  freehandStroke?: MotionPathVec2[];
  markDragging?: boolean;
};

export type PointerDownOutcome = PointerMoveOutcome & {
  selectedPointId?: string | null;
  capturePointer?: boolean;
  redraw?: boolean;
};

/** Below this, a place-gesture stays a corner (straight) click — matches pen-tool UX. */
const PLACE_CURVE_DRAG_THRESHOLD_PX = 6;
/** Min screen-space distance between freehand samples. */
const FREEHAND_SAMPLE_MIN_PX = 3;

export const IDLE_MOTION_PATH_GESTURE: MotionPathGesture = { kind: "idle" };

function pushEdit(points: MotionPathPoint[]): MotionPathPointsEdit {
  return { points, historyMode: "push" };
}

function replaceEdit(points: MotionPathPoint[]): MotionPathPointsEdit {
  return { points, historyMode: "replace" };
}

export function beginMotionPathPointerDown(params: {
  points: MotionPathPoint[];
  tool: MotionPathToolMode;
  selectedPointId: string | null;
  local: CanvasLocalPointer;
}): PointerDownOutcome {
  const { points, tool, selectedPointId, local } = params;
  const { px, py, width, height } = local;

  if (tool === "freehand") {
    const start = [{ x: clamp01(px / width), y: clamp01(py / height) }];
    return {
      gesture: { kind: "freehand", stroke: start },
      freehandStroke: start,
      selectedPointId: null,
      capturePointer: true,
    };
  }

  const hitPointIndex = hitTestPointIndex(points, px, py, width, height);
  const selectedIndex = selectedPointId ? points.findIndex((p) => p.id === selectedPointId) : -1;

  if (tool === "erase") {
    if (hitPointIndex < 0) return { gesture: IDLE_MOTION_PATH_GESTURE, selectedPointId: null };
    return {
      gesture: IDLE_MOTION_PATH_GESTURE, activeSegment: null, activeHandle: null, selectedPointId: null,
      pointsEdit: pushEdit(deleteMotionPathPoint(points, points[hitPointIndex].id)),
    };
  }

  const handleHit = hitTestSelectedPointHandle(points, selectedIndex, px, py, width, height);
  if (handleHit) {
    return {
      gesture: { kind: "dragHandle", hit: handleHit, hasHistory: false },
      activeHandle: handleHit,
      activeSegment:
        handleHit.side === "out" ? handleHit.pointIndex : Math.max(0, handleHit.pointIndex - 1),
      capturePointer: true,
      markDragging: true,
      redraw: true,
    };
  }

  if (tool === "convert") {
    if (hitPointIndex < 0) return { gesture: IDLE_MOTION_PATH_GESTURE, selectedPointId: null };
    return {
      gesture: IDLE_MOTION_PATH_GESTURE, activeSegment: null, activeHandle: null,
      selectedPointId: points[hitPointIndex].id,
      pointsEdit: pushEdit(toggleMotionPathPointSmooth(points, hitPointIndex)),
    };
  }

  if (hitPointIndex >= 0) {
    const id = points[hitPointIndex].id;
    return {
      gesture: { kind: "dragPoint", pointId: id, hasHistory: false },
      activeSegment: null,
      activeHandle: null,
      selectedPointId: id,
      capturePointer: true,
      markDragging: true,
      redraw: true,
    };
  }

  const segmentHit = hitTestSegment(points, px, py, width, height);
  if (segmentHit) {
    return {
      gesture: { kind: "dragCurve", segmentIndex: segmentHit.segmentIndex, t: segmentHit.t },
      selectedPointId: null,
      activeSegment: segmentHit.segmentIndex,
      activeHandle: null,
      capturePointer: true,
      markDragging: true,
      pointsEdit: pushEdit(
        setSegmentBend(
          points,
          segmentHit.segmentIndex,
          { x: px / width, y: py / height },
          segmentHit.t,
        ),
      ),
    };
  }

  if (points.length >= MOTION_PATH_MAX_POINTS) {
    return { gesture: IDLE_MOTION_PATH_GESTURE, selectedPointId: null };
  }

  const newPointId = randomUUID();
  const nextPoints = addMotionPathPoint(points, { id: newPointId, x: clamp01(px / width), y: clamp01(py / height) });
  const canDragCurve = points.length >= 1;
  const place: MotionPathGesture = {
    kind: "place", pointIndex: nextPoints.length - 1, originPx: px, originPy: py, curved: false,
  };
  return {
    gesture: canDragCurve ? place : IDLE_MOTION_PATH_GESTURE,
    activeSegment: null,
    activeHandle: null,
    selectedPointId: newPointId,
    pointsEdit: pushEdit(nextPoints),
    capturePointer: canDragCurve,
  };
}

export function stepMotionPathPointerMove(
  gesture: MotionPathGesture,
  points: MotionPathPoint[],
  local: CanvasLocalPointer,
): PointerMoveOutcome {
  const nx = clamp01(local.px / local.width);
  const ny = clamp01(local.py / local.height);
  const handleX = local.px / local.width;
  const handleY = local.py / local.height;

  switch (gesture.kind) {
    case "idle":
      return { gesture };
    case "freehand": {
      const last = gesture.stroke.at(-1);
      if (last) {
        const dx = (nx - last.x) * local.width;
        const dy = (ny - last.y) * local.height;
        if (Math.hypot(dx, dy) < FREEHAND_SAMPLE_MIN_PX) return { gesture };
      }
      const next = [...gesture.stroke, { x: nx, y: ny }];
      return { gesture: { kind: "freehand", stroke: next }, freehandStroke: next };
    }
    case "place": {
      if (!points[gesture.pointIndex]) return { gesture };
      const dragDist = Math.hypot(local.px - gesture.originPx, local.py - gesture.originPy);
      if (dragDist < PLACE_CURVE_DRAG_THRESHOLD_PX) {
        if (!gesture.curved) return { gesture };
        return {
          gesture: { ...gesture, curved: false },
          activeSegment: null,
          activeHandle: null,
          pointsEdit: replaceEdit(
            setMotionPathPointHandles(points, gesture.pointIndex, { handleIn: null, handleOut: null }),
          ),
        };
      }
      const dragOffsetX = (local.px - gesture.originPx) / local.width;
      const dragOffsetY = (local.py - gesture.originPy) / local.height;
      return {
        gesture: { ...gesture, curved: true },
        activeSegment: Math.max(0, gesture.pointIndex - 1),
        activeHandle: { pointIndex: gesture.pointIndex, side: "out" },
        markDragging: true,
        pointsEdit: replaceEdit(
          setMotionPathPointHandles(points, gesture.pointIndex, {
            handleIn: { x: -dragOffsetX, y: -dragOffsetY },
            handleOut: { x: dragOffsetX, y: dragOffsetY },
          }),
        ),
      };
    }
    case "dragHandle":
      return {
        gesture: { ...gesture, hasHistory: true },
        activeHandle: gesture.hit,
        pointsEdit: {
          points: moveMotionPathHandle(points, { ...gesture.hit, x: handleX, y: handleY }),
          historyMode: gesture.hasHistory ? "replace" : "push",
        },
      };
    case "dragPoint":
      return {
        gesture: { ...gesture, hasHistory: true },
        pointsEdit: {
          points: moveMotionPathPoint(points, gesture.pointId, nx, ny),
          historyMode: gesture.hasHistory ? "replace" : "push",
        },
      };
    case "dragCurve":
      return {
        gesture,
        pointsEdit: replaceEdit(
          setSegmentBend(points, gesture.segmentIndex, { x: handleX, y: handleY }, gesture.t),
        ),
      };
    default: {
      const unhandled: never = gesture;
      throw new Error(`Unhandled gesture: ${JSON.stringify(unhandled)}`);
    }
  }
}

export function finishMotionPathPointerUp(gesture: MotionPathGesture): {
  pointsEdit?: MotionPathPointsEdit;
  wasFreehand: boolean;
  wasDragging: boolean;
} {
  if (gesture.kind === "freehand") {
    const committed = commitFreehandStroke(gesture.stroke);
    return {
      wasFreehand: true,
      wasDragging: false,
      ...(committed ? { pointsEdit: pushEdit(committed) } : {}),
    };
  }
  return { wasFreehand: false, wasDragging: gesture.kind !== "idle" };
}

export function applyMotionPathDoubleClick(params: {
  points: MotionPathPoint[];
  tool: MotionPathToolMode;
  selectedPointId: string | null;
  local: CanvasLocalPointer;
}): { handled: boolean; points?: MotionPathPoint[]; selectedPointId?: string | null } {
  const { points, tool, selectedPointId, local } = params;
  if (tool === "erase" || tool === "convert" || tool === "freehand") return { handled: false };
  const { px, py, width, height } = local;
  const selectedIndex = selectedPointId ? points.findIndex((p) => p.id === selectedPointId) : -1;
  const handleHit = hitTestSelectedPointHandle(points, selectedIndex, px, py, width, height);
  if (handleHit) {
    return {
      handled: true,
      points: setMotionPathPointHandles(points, handleHit.pointIndex, {
        handleIn: handleHit.side === "in" ? null : undefined,
        handleOut: handleHit.side === "out" ? null : undefined,
      }),
    };
  }
  const hitPointIndex = hitTestPointIndex(points, px, py, width, height);
  if (hitPointIndex >= 0) {
    return { handled: true, selectedPointId: null, points: deleteMotionPathPoint(points, points[hitPointIndex].id) };
  }
  const segmentHit = hitTestSegment(points, px, py, width, height);
  if (segmentHit) {
    return { handled: true, points: clearSegmentBend(points, segmentHit.segmentIndex) };
  }
  return { handled: false };
}
