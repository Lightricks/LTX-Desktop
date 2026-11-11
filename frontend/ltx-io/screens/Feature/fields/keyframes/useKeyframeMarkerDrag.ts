import { type RefObject, useEffect, useRef, useState } from "react";
import type { PointerEvent } from "react";

import { type DraggedFrame, frameFromPointer } from "../../../../../lib/keyframe-timeline.ts";
import type { FeatureKeyframe } from "../../types";
import { type KeyframePatch, freeFrameFor } from "./keyframesModel";

/** A pointer that moves less than this many pixels is a click, not a drag. */
const DRAG_THRESHOLD_PX = 3;

/**
 * The drag of a marker along the rail. A press selects the marker. A move of a few pixels starts
 * the drag, which shows the marker at the free frame nearest the pointer. The release saves that
 * frame with `onUpdate`. A cancelled pointer saves nothing.
 * `frameAt` is the frame under a pointer, or null while the rail is not on screen.
 */
export function useKeyframeMarkerDrag({
  railRef,
  keyframes,
  lastFrame,
  disabled,
  onSelect,
  onUpdate,
  onDragFrame,
}: {
  railRef: RefObject<HTMLElement>;
  keyframes: readonly FeatureKeyframe[];
  lastFrame: number;
  disabled?: boolean;
  onSelect: (id: string) => void;
  onUpdate: (id: string, patch: KeyframePatch) => void;
  onDragFrame?: (drag: DraggedFrame | null) => void;
}): {
  drag: DraggedFrame | null;
  frameAt: (clientX: number) => number | null;
  startDrag: (event: PointerEvent<HTMLDivElement>, id: string) => void;
} {
  const [drag, setDrag] = useState<DraggedFrame | null>(null);
  // The drag ends in a window listener. It reads the last frame from here, not from a render.
  const dragRef = useRef<DraggedFrame | null>(null);
  const pointerStart = useRef<{ id: string; x: number; moved: boolean } | null>(null);
  // A drag ends in a window listener, after later renders. It reads the newest props from here.
  const latest = useRef({ keyframes, lastFrame, onUpdate, onDragFrame });
  useEffect(() => {
    latest.current = { keyframes, lastFrame, onUpdate, onDragFrame };
  });
  const detachDrag = useRef<(() => void) | null>(null);
  // Leave no window listener behind when the strip goes away in the middle of a drag.
  useEffect(() => () => detachDrag.current?.(), []);

  const frameAt = (clientX: number): number | null => {
    const rail = railRef.current;
    if (!rail) return null;
    return frameFromPointer(clientX, rail.getBoundingClientRect(), latest.current.lastFrame);
  };

  const startDrag = (event: PointerEvent<HTMLDivElement>, id: string) => {
    if (disabled) return;
    // Stop the browser from selecting text or starting a native drag of the image.
    event.preventDefault();
    event.currentTarget.focus();
    onSelect(id);
    const pressed = latest.current.keyframes.find((keyframe) => keyframe.id === id);
    if (pressed) onDragFrame?.({ id, frameIndex: pressed.frameIndex });
    pointerStart.current = { id, x: event.clientX, moved: false };

    const move = (moveEvent: globalThis.PointerEvent) => {
      const start = pointerStart.current;
      if (!start) return;
      if (!start.moved && Math.abs(moveEvent.clientX - start.x) < DRAG_THRESHOLD_PX) return;
      start.moved = true;
      const target = frameAt(moveEvent.clientX);
      if (target === null) return;
      const { keyframes: current, lastFrame: last } = latest.current;
      const frameIndex = freeFrameFor(current, id, target, last);
      if (frameIndex === null) return;
      dragRef.current = { id, frameIndex };
      setDrag(dragRef.current);
      latest.current.onDragFrame?.(dragRef.current);
    };
    const finish = (upEvent: globalThis.PointerEvent) => {
      const start = pointerStart.current;
      pointerStart.current = null;
      detach();
      const dropped = dragRef.current;
      dragRef.current = null;
      setDrag(null);
      latest.current.onDragFrame?.(null);
      if (dropped && start?.moved && upEvent.type === "pointerup") {
        // An import can finish during the drag and take the frame. Check the list as it is now.
        const { keyframes: current, lastFrame: last, onUpdate: update } = latest.current;
        const frameIndex = freeFrameFor(current, dropped.id, dropped.frameIndex, last);
        if (frameIndex !== null) update(dropped.id, { frameIndex });
      }
    };
    const detach = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
      detachDrag.current = null;
    };
    detachDrag.current = detach;
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", finish);
  };

  return { drag, frameAt, startDrag };
}
