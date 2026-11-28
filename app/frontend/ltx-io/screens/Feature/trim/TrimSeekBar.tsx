import { clsx } from "clsx";
import { useCallback, useEffect, useLayoutEffect, useRef } from "react";

import { PlayheadNeedle } from "./PlayheadNeedle.tsx";
import {
  PLAYHEAD_HEAD_HEIGHT_PX,
  SELECTION_BORDER_PX,
} from "./trimConstants.ts";
import { pixelsToTime, timeToPixels } from "./trimGeometry.ts";
import {
  createPlayheadLoop,
  placePlayhead,
  snapPlayheadPx,
  type PlayheadLoop,
  type SelectionRect,
} from "./trimPlayhead.ts";
import styles from "./TrimSeekBar.module.scss";

/** Movement below this stays a click. Past it, the press is a drag. */
const PLAYHEAD_CLICK_SLOP_PX = 3;

type SeekGeometry = {
  pixelsPerSecond: number;
  leftLimit: number;
  rightLimit: number;
  inPx: number;
  outPx: number;
  selection: SelectionRect;
};

/**
 * Playhead over the trim track. Ported from LTX.io `TrimSeekBar`.
 * Scrubbing is clamped to `startSec`–`endSec` (selection for A2V trim, full
 * clip for video). When that window is wider than the In/Out selection, a
 * drag sticks to In/Out while close, as in LTX.io's video trimmer.
 */
export function TrimSeekBar({
  mediaRef,
  pixelsPerSecond,
  clipWidth,
  clipHeight,
  startSec,
  endSec,
  selectionStartSec,
  selectionEndSec,
  selection,
  isPlaying,
  onSeek,
  onSnapSelection,
}: {
  mediaRef: React.RefObject<HTMLMediaElement>;
  pixelsPerSecond: number;
  clipWidth: number;
  clipHeight: number;
  startSec: number;
  endSec: number;
  selectionStartSec: number;
  selectionEndSec: number;
  /** Laid-out In/Out box, so the stem yields to the handles where they really are. */
  selection: SelectionRect;
  isPlaying: boolean;
  onSeek?: (seconds: number) => void;
  /**
   * Video trim only. A double-click moves the selection so it starts at the
   * needle. Omitted for audio, whose playhead never leaves the selection.
   */
  onSnapSelection?: (needleSec: number) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const handleRef = useRef<HTMLDivElement>(null);
  const stemRef = useRef<HTMLDivElement>(null);
  const isDraggingRef = useRef(false);
  const loopRef = useRef<PlayheadLoop | null>(null);
  const pressXRef = useRef<number | null>(null);
  /** Time at click.detail 1. A drag clears it so the second click cannot snap. */
  const snapTimeRef = useRef<number | null>(null);
  const snapDraggedRef = useRef(false);

  const geometry: SeekGeometry = {
    pixelsPerSecond,
    leftLimit: timeToPixels(startSec, pixelsPerSecond),
    rightLimit: timeToPixels(endSec, pixelsPerSecond),
    inPx: timeToPixels(selectionStartSec, pixelsPerSecond),
    outPx: timeToPixels(selectionEndSec, pixelsPerSecond),
    selection,
  };
  // Read by the rAF loop, which outlives renders: In/Out drags must not
  // tear down and rebuild the media listeners.
  const geometryRef = useRef(geometry);

  const paint = useCallback((timeSec: number) => {
    const handle = handleRef.current;
    const stem = stemRef.current;
    if (!handle || !stem) return;
    const { leftLimit, rightLimit, selection: rect } = geometryRef.current;
    const raw = timeToPixels(timeSec, geometryRef.current.pixelsPerSecond);
    placePlayhead(
      handle,
      stem,
      Math.max(leftLimit, Math.min(raw, rightLimit)),
      rect,
    );
  }, []);

  const { leftLimit, rightLimit, inPx, outPx } = geometry;
  useLayoutEffect(() => {
    geometryRef.current = {
      pixelsPerSecond,
      leftLimit,
      rightLimit,
      inPx,
      outPx,
      selection: { left: selection.left, width: selection.width },
    };
    const media = mediaRef.current;
    if (media && !isDraggingRef.current) paint(media.currentTime);
  }, [
    mediaRef,
    paint,
    pixelsPerSecond,
    leftLimit,
    rightLimit,
    inPx,
    outPx,
    selection.left,
    selection.width,
  ]);

  // Follow playback with rAF while playing, and settle on the exact frame on
  // pause/seek. Writing `style.left` directly keeps this off the React path.
  useEffect(() => {
    const media = mediaRef.current;
    if (!media) return;

    const loop = createPlayheadLoop({
      media,
      paint,
      frames: {
        request: (callback) => requestAnimationFrame(callback),
        cancel: (id) => cancelAnimationFrame(id),
      },
    });
    if (isDraggingRef.current) loop.beginScrub();
    loopRef.current = loop;

    const detach = loop.attach();
    return () => {
      detach();
      loopRef.current = null;
    };
  }, [mediaRef, paint]);

  const seekToClientX = useCallback(
    (clientX: number) => {
      const container = containerRef.current;
      const media = mediaRef.current;
      const handle = handleRef.current;
      const stem = stemRef.current;
      if (!container || !media || !handle || !stem) return;

      const current = geometryRef.current;
      const rect = container.getBoundingClientRect();
      const clamped = Math.max(
        current.leftLimit,
        Math.min(clientX - rect.left, current.rightLimit),
      );
      const snapsToSelection =
        current.leftLimit < current.inPx || current.rightLimit > current.outPx;
      const px = snapsToSelection
        ? snapPlayheadPx(clamped, current.inPx, current.outPx)
        : clamped;
      placePlayhead(handle, stem, px, current.selection);
      const timeSec = pixelsToTime(px, current.pixelsPerSecond);
      media.currentTime = timeSec;
      onSeek?.(timeSec);
    },
    [mediaRef, onSeek],
  );

  const endScrub = () => {
    isDraggingRef.current = false;
    loopRef.current?.endScrub();
  };

  // The tip overlaps the stroke. The needle paints on top of the frame.
  const needleHeight =
    clipHeight + PLAYHEAD_HEAD_HEIGHT_PX - SELECTION_BORDER_PX;

  return (
    <div
      ref={containerRef}
      className={styles.container}
      style={{ width: clipWidth }}
    >
      <div
        ref={handleRef}
        className={clsx(styles.handle, isPlaying && styles.playing)}
        style={{
          top: -(PLAYHEAD_HEAD_HEIGHT_PX - SELECTION_BORDER_PX),
          height: needleHeight,
        }}
        aria-label="Playhead"
        title={
          onSnapSelection
            ? "Double-click to start the selection here"
            : undefined
        }
        onPointerDown={(event) => {
          event.preventDefault();
          event.stopPropagation();
          isDraggingRef.current = true;
          loopRef.current?.beginScrub();
          (event.target as Element).setPointerCapture(event.pointerId);
          if (onSnapSelection) {
            pressXRef.current = event.clientX;
            snapDraggedRef.current = false;
          }
          // Seeking on pointerdown would jump the needle to the click inside
          // the 13px head, and the second click of a double-click would miss.
          if (!onSnapSelection) seekToClientX(event.clientX);
        }}
        onPointerMove={(event) => {
          if (!isDraggingRef.current) return;
          if (
            pressXRef.current != null &&
            Math.abs(event.clientX - pressXRef.current) <
              PLAYHEAD_CLICK_SLOP_PX
          ) {
            return;
          }
          snapTimeRef.current = null;
          snapDraggedRef.current = true;
          seekToClientX(event.clientX);
        }}
        onPointerUp={endScrub}
        onPointerCancel={() => {
          pressXRef.current = null;
          snapTimeRef.current = null;
          snapDraggedRef.current = false;
          endScrub();
        }}
        onClick={(event) => {
          // detail 1 is the start of the browser's double-click sequence.
          // detail 2 must keep that time; the platform interval is not ours.
          if (
            !onSnapSelection ||
            snapDraggedRef.current ||
            event.detail !== 1
          ) {
            return;
          }
          snapTimeRef.current = mediaRef.current?.currentTime ?? 0;
        }}
        onDoubleClick={(event) => {
          const time = snapTimeRef.current;
          snapTimeRef.current = null;
          if (time == null || !onSnapSelection) return;
          event.preventDefault();
          event.stopPropagation();
          const media = mediaRef.current;
          if (media && media.currentTime !== time) {
            media.currentTime = time;
            onSeek?.(time);
          }
          onSnapSelection(time);
        }}
      >
        <PlayheadNeedle height={needleHeight} />
        <div
          className={styles.grip}
          style={{ height: PLAYHEAD_HEAD_HEIGHT_PX }}
        />
        <div
          ref={stemRef}
          className={styles.stem}
          style={{ top: PLAYHEAD_HEAD_HEIGHT_PX }}
        />
      </div>
    </div>
  );
}
