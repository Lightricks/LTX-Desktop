import { useRef, type RefObject } from "react";

import {
  clampTrimRange,
  pixelsToTime,
  type TrimRange,
} from "./trimGeometry.ts";

type DragMode = "start" | "end" | "both";

const KEYBOARD_STEP_SECONDS = 0.5;

/**
 * Pointer drags for In, Out, and the selection body, plus arrow-key nudges for
 * the In/Out sliders. Every change goes through `clampTrimRange` so the window
 * never leaves `[min, max]`.
 */
export function useTrimRangeDrag({
  containerRef,
  pixelsPerSecond,
  durationSeconds,
  maxDurationSeconds,
  minDurationSeconds,
  range,
  onRangeChange,
}: {
  containerRef: RefObject<HTMLDivElement | null>;
  pixelsPerSecond: number;
  durationSeconds: number;
  maxDurationSeconds: number;
  minDurationSeconds: number;
  range: TrimRange;
  onRangeChange: (range: TrimRange) => void;
}) {
  const dragRef = useRef<{ mode: DragMode; grabOffsetSec: number } | null>(null);

  const clamp = (next: TrimRange, anchor: DragMode) =>
    clampTrimRange(next, {
      durationSeconds,
      maxDurationSeconds,
      minDurationSeconds,
      anchor,
    });

  const secondsAtClientX = (clientX: number) => {
    const container = containerRef.current;
    if (!container || pixelsPerSecond === 0) return 0;
    const rect = container.getBoundingClientRect();
    return pixelsToTime(clientX - rect.left, pixelsPerSecond);
  };

  const beginDrag = (event: React.PointerEvent<HTMLElement>, mode: DragMode) => {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      mode,
      grabOffsetSec:
        mode === "both" ? secondsAtClientX(event.clientX) - range.startSec : 0,
    };
  };

  const continueDrag = (event: React.PointerEvent<HTMLElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const seconds = secondsAtClientX(event.clientX);

    if (drag.mode === "start") {
      onRangeChange(clamp({ startSec: seconds, endSec: range.endSec }, "start"));
      return;
    }
    if (drag.mode === "end") {
      onRangeChange(clamp({ startSec: range.startSec, endSec: seconds }, "end"));
      return;
    }

    const width = range.endSec - range.startSec;
    const startSec = seconds - drag.grabOffsetSec;
    onRangeChange(clamp({ startSec, endSec: startSec + width }, "both"));
  };

  const endDrag = (event: React.PointerEvent<HTMLElement>) => {
    if (!dragRef.current) return;
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const dragHandlers = (mode: DragMode) => ({
    onPointerDown: (event: React.PointerEvent<HTMLElement>) =>
      beginDrag(event, mode),
    onPointerMove: continueDrag,
    onPointerUp: endDrag,
    onPointerCancel: endDrag,
  });

  const keyHandler = (edge: "start" | "end") => (event: React.KeyboardEvent) => {
    const direction =
      event.key === "ArrowRight" || event.key === "ArrowUp"
        ? 1
        : event.key === "ArrowLeft" || event.key === "ArrowDown"
          ? -1
          : 0;
    if (direction === 0) return;
    event.preventDefault();
    const step = direction * KEYBOARD_STEP_SECONDS;
    onRangeChange(
      clamp(
        edge === "start"
          ? { startSec: range.startSec + step, endSec: range.endSec }
          : { startSec: range.startSec, endSec: range.endSec + step },
        edge,
      ),
    );
  };

  return { dragHandlers, keyHandler };
}
