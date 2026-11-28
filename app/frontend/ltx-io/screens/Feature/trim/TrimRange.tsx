import { clsx } from "clsx";
import { useEffect, useRef, useState, type ReactNode } from "react";

import { TrimSeekBar } from "./TrimSeekBar.tsx";
import {
  formatTrimTimecode,
  pixelsToTime,
  playheadWindow,
  snapSelectionToNeedle,
  timeToPixels,
  trimMediaWidth,
  trimSelectionRect,
  type TrimRange as TrimRangeValue,
} from "./trimGeometry.ts";
import {
  CLIP_HEIGHT_AUDIO_COMPACT,
  MIN_TRIM_DURATION_SECONDS,
  PLAYHEAD_HEAD_HEIGHT_PX,
  RESIZE_HANDLE_WIDTH,
  SELECTION_BORDER_PX,
  TRIMMER_CONTAINER_WIDTH,
} from "./trimConstants.ts";
import styles from "./TrimRange.module.scss";
import { useTrimRangeDrag } from "./useTrimRangeDrag.ts";

/** Where the selected-duration label sits relative to the selection. */
export type TrimLabelPlacement = "inside" | "top" | "bottom";

/** Below this selection width an `inside` label is hidden rather than overlapping In/Out (LTX.io). */
const MIN_WIDTH_FOR_INSIDE_LABEL_PX = 40;

const LABEL_PLACEMENT_CLASS: Record<TrimLabelPlacement, string | undefined> = {
  inside: styles.labelInside,
  top: styles.labelTop,
  bottom: styles.labelBottom,
};

/**
 * Ported from LTX.io `TrimRange` + `RangeSelector`, with react-rnd replaced by
 * pointer capture so Desktop does not pull in a drag-and-resize dependency.
 *
 * Geometry: the filmstrip fills the bar. In/Out handles overlay the cut
 * frames so the left/right of the selection is not an empty gutter. A2V's
 * audio-trim modal shares this bar — the overlay handles are the blessed
 * look for both surfaces.
 */
export function TrimRange({
  durationSeconds,
  maxDurationSeconds,
  minDurationSeconds = MIN_TRIM_DURATION_SECONDS,
  range,
  onRangeChange,
  mediaRef,
  isPlaying,
  onSeek,
  playheadRange,
  clipHeight = CLIP_HEIGHT_AUDIO_COMPACT,
  labelPlacement = "inside",
  children,
}: {
  durationSeconds: number;
  maxDurationSeconds: number;
  minDurationSeconds?: number;
  range: TrimRangeValue;
  onRangeChange: (range: TrimRangeValue) => void;
  mediaRef: React.RefObject<HTMLMediaElement>;
  isPlaying: boolean;
  onSeek?: (seconds: number) => void;
  /**
   * Where the playhead may scrub. Defaults to the In/Out selection (A2V); the
   * video trimmer passes the full clip so the playhead can leave the selection.
   */
  playheadRange?: TrimRangeValue;
  clipHeight?: number;
  /** `top`/`bottom` keep the label readable however narrow the selection gets. */
  labelPlacement?: TrimLabelPlacement;
  children: (dimensions: { clipWidth: number; clipHeight: number }) => ReactNode;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(TRIMMER_CONTAINER_WIDTH);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const clipWidth = trimMediaWidth(containerWidth);
  const pixelsPerSecond = durationSeconds > 0 ? clipWidth / durationSeconds : 0;

  const startPx = timeToPixels(range.startSec, pixelsPerSecond);
  const endPx = timeToPixels(range.endSec, pixelsPerSecond);
  const selection = trimSelectionRect(
    startPx,
    endPx,
    RESIZE_HANDLE_WIDTH,
    containerWidth,
  );
  const { dragHandlers, keyHandler } = useTrimRangeDrag({
    containerRef,
    pixelsPerSecond,
    durationSeconds,
    maxDurationSeconds,
    minDurationSeconds,
    range,
    onRangeChange,
  });

  const selectedSeconds = range.endSec - range.startSec;
  const playhead = playheadWindow(playheadRange, range);
  const snapSelection =
    playheadRange == null
      ? undefined
      : (needleSec: number) => {
          const next = snapSelectionToNeedle(needleSec, range, {
            durationSeconds,
            maxDurationSeconds,
            minDurationSeconds,
          });
          if (
            next == null ||
            (next.startSec === range.startSec && next.endSec === range.endSec)
          ) {
            return;
          }
          const media = mediaRef.current;
          if (media && media.currentTime !== needleSec) {
            media.currentTime = needleSec;
            onSeek?.(needleSec);
          }
          onRangeChange(next);
        };
  const snapFromTrack = (event: React.MouseEvent<HTMLDivElement>) => {
    if (snapSelection == null || pixelsPerSecond <= 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const time = pixelsToTime(event.clientX - rect.left, pixelsPerSecond);
    // The selection body is a drag. Only the filmstrip outside it snaps.
    if (time >= range.startSec && time <= range.endSec) return;
    snapSelection(time);
  };
  const showLabel =
    labelPlacement !== "inside" ||
    endPx - startPx >= MIN_WIDTH_FOR_INSIDE_LABEL_PX;

  return (
    <div
      ref={containerRef}
      className={clsx(
        styles.container,
        labelPlacement === "top" && styles.labelSpaceTop,
        labelPlacement === "bottom" && styles.labelSpaceBottom,
      )}
      style={
        {
          height: clipHeight,
          "--playhead-head": `${PLAYHEAD_HEAD_HEIGHT_PX}px`,
          "--selection-border": `${SELECTION_BORDER_PX}px`,
        } as React.CSSProperties
      }
    >
      <div
        className={styles.track}
        style={{ left: 0, width: clipWidth }}
        title={
          snapSelection ? "Double-click to start the selection here" : undefined
        }
        onDoubleClick={snapSelection ? snapFromTrack : undefined}
      >
        {children({ clipWidth, clipHeight })}
        <div
          className={styles.dimmed}
          style={{ left: 0, width: startPx }}
          aria-hidden
        />
        <div
          className={styles.dimmed}
          style={{
            left: endPx,
            width: Math.max(0, clipWidth - endPx),
          }}
          aria-hidden
        />
      </div>

      <div
        className={styles.selection}
        style={{ left: selection.left, width: selection.width }}
        {...dragHandlers("both")}
      >
        <span
          className={clsx(styles.handle, styles.handleLeft)}
          style={{ width: RESIZE_HANDLE_WIDTH }}
          role="slider"
          aria-label="Trim start"
          aria-valuemin={0}
          aria-valuemax={durationSeconds}
          aria-valuenow={range.startSec}
          aria-valuetext={formatTrimTimecode(range.startSec)}
          tabIndex={0}
          onKeyDown={keyHandler("start")}
          {...dragHandlers("start")}
        />
        <span
          className={clsx(styles.handle, styles.handleRight)}
          style={{ width: RESIZE_HANDLE_WIDTH }}
          role="slider"
          aria-label="Trim end"
          aria-valuemin={0}
          aria-valuemax={durationSeconds}
          aria-valuenow={range.endSec}
          aria-valuetext={formatTrimTimecode(range.endSec)}
          tabIndex={0}
          onKeyDown={keyHandler("end")}
          {...dragHandlers("end")}
        />
      </div>

      {showLabel ? (
        <span
          className={clsx(styles.label, LABEL_PLACEMENT_CLASS[labelPlacement])}
          style={{ left: selection.left + selection.width / 2 }}
          aria-hidden
        >
          {formatTrimTimecode(selectedSeconds)}
        </span>
      ) : null}

      {/* Above the frame so the needle stays visible over the stroke and
          the brackets. The stem still yields its hits to In/Out. */}
      {pixelsPerSecond > 0 ? (
        <div
          className={styles.seekLayer}
          style={{ left: 0, width: clipWidth }}
        >
          <TrimSeekBar
            mediaRef={mediaRef}
            pixelsPerSecond={pixelsPerSecond}
            clipWidth={clipWidth}
            clipHeight={clipHeight}
            startSec={playhead.startSec}
            endSec={playhead.endSec}
            selectionStartSec={range.startSec}
            selectionEndSec={range.endSec}
            selection={selection}
            isPlaying={isPlaying}
            onSeek={onSeek}
            onSnapSelection={snapSelection}
          />
        </div>
      ) : null}

      <span className={styles.srOnly} role="status">
        {`Selected ${formatTrimTimecode(selectedSeconds)} starting at ${formatTrimTimecode(range.startSec)}`}
      </span>
    </div>
  );
}
