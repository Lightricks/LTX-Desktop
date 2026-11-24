import { type MutableRefObject, type RefObject, useCallback, useEffect, useRef } from "react";

import type { MotionPathPoint, MotionPathVec2 } from "../../../lib/motionPathSampler";
import type { MotionPathTool } from "./MotionPathCanvas";
import { getCanvasPointerLocal } from "./motionPathCanvasUtils";
import {
  applyMotionPathDoubleClick,
  beginMotionPathPointerDown,
  finishMotionPathPointerUp,
  IDLE_MOTION_PATH_GESTURE,
  type MotionPathGesture,
  stepMotionPathPointerMove,
} from "./motionPathGestures";
import type { HandleHit } from "./motionPathHitTest";

export type UseMotionPathPointerHandlersOptions = {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  points: MotionPathPoint[];
  tool: MotionPathTool;
  onPointsChange: (points: MotionPathPoint[], historyMode?: "push" | "replace") => void;
  selectedPointId: string | null;
  setSelectedPointId: (id: string | null) => void;
  setFreehandStroke: (stroke: MotionPathVec2[]) => void;
  activeSegmentRef: MutableRefObject<number | null>;
  activeHandleRef: MutableRefObject<HandleHit | null>;
  draw: () => void;
};

export type UseMotionPathPointerHandlersResult = {
  handlePointerDown: (event: React.PointerEvent<HTMLCanvasElement>) => void;
  handlePointerMove: (event: React.PointerEvent<HTMLCanvasElement>) => void;
  handlePointerUp: (event: React.PointerEvent<HTMLCanvasElement>) => void;
  handleDoubleClick: (event: React.MouseEvent<HTMLCanvasElement>) => void;
};

export function useMotionPathPointerHandlers({
  canvasRef,
  points,
  tool,
  onPointsChange,
  selectedPointId,
  setSelectedPointId,
  setFreehandStroke,
  activeSegmentRef,
  activeHandleRef,
  draw,
}: UseMotionPathPointerHandlersOptions): UseMotionPathPointerHandlersResult {
  const gestureRef = useRef<MotionPathGesture>(IDLE_MOTION_PATH_GESTURE);
  const pointsRef = useRef(points);
  pointsRef.current = points;
  const onPointsChangeRef = useRef(onPointsChange);
  onPointsChangeRef.current = onPointsChange;

  useEffect(() => {
    if (tool !== "freehand" && gestureRef.current.kind === "freehand") {
      gestureRef.current = IDLE_MOTION_PATH_GESTURE;
      setFreehandStroke([]);
    }
  }, [setFreehandStroke, tool]);

  const pointerLocal = useCallback(
    (event: React.PointerEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      return getCanvasPointerLocal(canvas, event.clientX, event.clientY);
    },
    [canvasRef],
  );

  const handlePointerDown = useCallback(
    (event: React.PointerEvent<HTMLCanvasElement>) => {
      const local = pointerLocal(event);
      if (!local) return;
      const outcome = beginMotionPathPointerDown({
        points,
        tool,
        selectedPointId,
        local,
      });
      gestureRef.current = outcome.gesture;
      if (outcome.activeSegment !== undefined) {
        activeSegmentRef.current = outcome.activeSegment;
      }
      if (outcome.activeHandle !== undefined) {
        activeHandleRef.current = outcome.activeHandle;
      }
      if (outcome.selectedPointId !== undefined) {
        setSelectedPointId(outcome.selectedPointId);
      }
      if (outcome.freehandStroke !== undefined) {
        setFreehandStroke(outcome.freehandStroke);
      }
      if (outcome.pointsEdit) {
        onPointsChange(outcome.pointsEdit.points, outcome.pointsEdit.historyMode);
      }
      if (outcome.capturePointer) {
        event.currentTarget.setPointerCapture(event.pointerId);
      }
      if (outcome.markDragging) {
        event.currentTarget.dataset.dragging = "true";
      }
      if (outcome.redraw) {
        draw();
      }
    },
    [
      activeHandleRef,
      activeSegmentRef,
      draw,
      onPointsChange,
      pointerLocal,
      points,
      selectedPointId,
      setFreehandStroke,
      setSelectedPointId,
      tool,
    ],
  );

  const handlePointerMove = useCallback(
    (event: React.PointerEvent<HTMLCanvasElement>) => {
      const local = pointerLocal(event);
      if (!local) return;
      const outcome = stepMotionPathPointerMove(gestureRef.current, pointsRef.current, local);
      gestureRef.current = outcome.gesture;
      if (outcome.activeSegment !== undefined) {
        activeSegmentRef.current = outcome.activeSegment;
      }
      if (outcome.activeHandle !== undefined) {
        activeHandleRef.current = outcome.activeHandle;
      }
      if (outcome.freehandStroke !== undefined) {
        setFreehandStroke(outcome.freehandStroke);
      }
      if (outcome.markDragging) {
        event.currentTarget.dataset.dragging = "true";
      }
      if (outcome.pointsEdit) {
        onPointsChangeRef.current(outcome.pointsEdit.points, outcome.pointsEdit.historyMode);
      }
    },
    [activeHandleRef, activeSegmentRef, pointerLocal, setFreehandStroke],
  );

  const handlePointerUp = useCallback(
    (event: React.PointerEvent<HTMLCanvasElement>) => {
      const outcome = finishMotionPathPointerUp(gestureRef.current);
      gestureRef.current = IDLE_MOTION_PATH_GESTURE;
      activeHandleRef.current = null;
      delete event.currentTarget.dataset.dragging;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
      if (outcome.wasFreehand) {
        setFreehandStroke([]);
        if (outcome.pointsEdit) {
          onPointsChangeRef.current(outcome.pointsEdit.points, outcome.pointsEdit.historyMode);
        }
        draw();
        return;
      }
      if (outcome.wasDragging) {
        activeSegmentRef.current = null;
        draw();
      }
    },
    [activeHandleRef, activeSegmentRef, draw, setFreehandStroke],
  );

  const handleDoubleClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const local = getCanvasPointerLocal(canvas, event.clientX, event.clientY);
      if (!local) return;
      const outcome = applyMotionPathDoubleClick({
        points,
        tool,
        selectedPointId,
        local,
      });
      if (!outcome.handled) return;
      event.preventDefault();
      if (outcome.selectedPointId !== undefined) {
        setSelectedPointId(outcome.selectedPointId);
      }
      if (outcome.points) {
        onPointsChange(outcome.points);
      }
    },
    [canvasRef, onPointsChange, points, selectedPointId, setSelectedPointId, tool],
  );

  return {
    handlePointerDown,
    handlePointerMove,
    handlePointerUp,
    handleDoubleClick,
  };
}
