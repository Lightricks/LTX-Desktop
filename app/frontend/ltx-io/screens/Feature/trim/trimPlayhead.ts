/**
 * Drives the trim playhead's requestAnimationFrame loop from media events.
 *
 * Extracted from `TrimSeekBar` so the scrub-during-playback path is testable
 * without a DOM: scrubbing seeks the media, and a `seeked` event that arrives
 * while playback continues must not be mistaken for a pause.
 */

import {
  PLAYHEAD_HANDLE_WIDTH_PX,
  RESIZE_HANDLE_WIDTH,
} from "./trimConstants.ts";

export type PlayheadMedia = {
  readonly currentTime: number;
  readonly paused: boolean;
  addEventListener: (type: string, listener: () => void) => void;
  removeEventListener: (type: string, listener: () => void) => void;
};

export type PlayheadFrames = {
  request: (callback: () => void) => number;
  cancel: (handle: number) => void;
};

export type PlayheadLoop = {
  /** Subscribes to media events and starts the loop when already playing. */
  attach: () => () => void;
  /** Suppresses painting while the user drags the handle. */
  beginScrub: () => void;
  /** Resumes painting, restarting the loop if playback never stopped. */
  endScrub: () => void;
};

export type PlayheadPointerEvents = "auto" | "none";

/** The In/Out box as laid out: In fills its left edge, Out its right. */
export type SelectionRect = { left: number; width: number };

/**
 * Hit-testing for the playhead stem, which is 13px centered on its time. When
 * it covers In or Out those handles must win; the head above the track never
 * overlaps them, so the playhead stays grabbable there.
 *
 * Coordinates are seek-layer pixels, 0 at the clip's left edge. Handles are
 * read from the clamped selection rect, so at the clip edges In sits inside
 * `[0, handleWidth]` rather than left of the In time.
 */
export function playheadPointerEvents(
  playheadPx: number,
  selection: SelectionRect,
  options?: {
    handleWidthPx?: number;
    playheadWidthPx?: number;
  },
): PlayheadPointerEvents {
  const handleWidth = options?.handleWidthPx ?? RESIZE_HANDLE_WIDTH;
  const half = (options?.playheadWidthPx ?? PLAYHEAD_HANDLE_WIDTH_PX) / 2;
  const left = playheadPx - half;
  const right = playheadPx + half;
  const selectionRight = selection.left + selection.width;

  const overlapsIn = left < selection.left + handleWidth && right > selection.left;
  const overlapsOut = right > selectionRight - handleWidth && left < selectionRight;
  return overlapsIn || overlapsOut ? "none" : "auto";
}

/** Magnetic grab distance while dragging the playhead past In/Out. */
export const PLAYHEAD_SNAP_PX = 10;

/**
 * Ported from LTX.io `snapVideoPlayheadX`: a dragged playhead sticks to In or
 * Out when close, then unsticks once the pointer leaves the threshold. The
 * magnet never exceeds half the selection minus 1px, so a narrow selection
 * keeps an interior strip the playhead can sit in.
 */
export function snapPlayheadPx(
  px: number,
  inPx: number,
  outPx: number,
  thresholdPx: number = PLAYHEAD_SNAP_PX,
): number {
  const effective = Math.min(
    thresholdPx,
    Math.max(0, Math.abs(outPx - inPx) / 2 - 1),
  );
  const distIn = Math.abs(px - inPx);
  const distOut = Math.abs(px - outPx);
  if (distIn <= effective && distIn <= distOut) return inPx;
  if (distOut <= effective) return outPx;
  return px;
}

/** Writes the playhead's left edge and the stem's pointer-events. */
export function placePlayhead(
  handle: { style: { left: string } },
  stem: { style: { pointerEvents: string } },
  playheadPx: number,
  selection: SelectionRect,
): void {
  handle.style.left = `${playheadPx}px`;
  stem.style.pointerEvents = playheadPointerEvents(playheadPx, selection);
}

export function createPlayheadLoop({
  media,
  paint,
  frames,
}: {
  media: PlayheadMedia;
  paint: (timeSec: number) => void;
  frames: PlayheadFrames;
}): PlayheadLoop {
  let frameHandle: number | null = null;
  let isScrubbing = false;

  const tick = () => {
    // Keep the loop alive across a drag: the pointer owns the handle position
    // until pointerup, but playback never stopped.
    if (!isScrubbing) paint(media.currentTime);
    frameHandle = frames.request(tick);
  };

  const start = () => {
    if (frameHandle === null) frameHandle = frames.request(tick);
  };

  const stop = () => {
    if (frameHandle !== null) {
      frames.cancel(frameHandle);
      frameHandle = null;
    }
  };

  const onPause = () => {
    stop();
    paint(media.currentTime);
  };

  const onSeeked = () => {
    if (!isScrubbing) paint(media.currentTime);
    if (media.paused) stop();
    else start();
  };

  return {
    attach: () => {
      media.addEventListener("play", start);
      media.addEventListener("pause", onPause);
      media.addEventListener("seeked", onSeeked);
      paint(media.currentTime);
      if (!media.paused) start();
      return () => {
        media.removeEventListener("play", start);
        media.removeEventListener("pause", onPause);
        media.removeEventListener("seeked", onSeeked);
        stop();
      };
    },
    beginScrub: () => {
      isScrubbing = true;
    },
    endScrub: () => {
      isScrubbing = false;
      if (media.paused) paint(media.currentTime);
      else start();
    },
  };
}
