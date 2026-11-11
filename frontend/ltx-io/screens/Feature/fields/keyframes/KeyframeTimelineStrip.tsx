import { clsx } from "clsx";
import { useRef, useState } from "react";
import type { CSSProperties, DragEvent, KeyboardEvent } from "react";

import {
  type DraggedFrame,
  rulerTicks,
  withDraggedFrame,
} from "../../../../../lib/keyframe-timeline.ts";
import type { FeatureKeyframe } from "../../types";
import { KeyframeMarker } from "./KeyframeMarker";
import { KEYFRAME_EDIT_KEYS, type KeyframePatch, byFrame, keyframePatchForKey } from "./keyframesModel";
import { useKeyframeMarkerDrag } from "./useKeyframeMarkerDrag";
import styles from "./KeyframeTimelineStrip.module.scss";

/**
 * A timeline of the stills, with a ruler in seconds under it. Each marker shows the place of
 * its still in time order, like its list row. Click a marker to select it. Drag it to move it.
 * The arrow keys move it by one frame (left, right) or change its strength (up, down).
 * With `onSeek`, a click on the empty track moves the playhead. Without it, the track has no
 * playhead and a click on it does nothing.
 * Files dropped on the track go to the frame under the pointer. A marker that gets focus is selected.
 * The size of the markers comes from the field, which makes them smaller as stills are added.
 */
export function KeyframeTimelineStrip({
  keyframes,
  lastFrame,
  fps,
  selectedId,
  playhead,
  markerSize,
  railHeight,
  disabled,
  onSelect,
  onUpdate,
  onSeek,
  onDragFrame,
  onDropFiles,
  onRemove,
}: {
  keyframes: readonly FeatureKeyframe[];
  lastFrame: number;
  fps: number;
  selectedId: string | null;
  /** The frame of the playhead. Not set: no playhead. */
  playhead?: number;
  /** The side of a marker, in pixels. It never makes the strip higher or lower. */
  markerSize: number;
  /** The height of the rail, in pixels. The default fits the biggest field marker. */
  railHeight?: number;
  disabled?: boolean;
  onSelect: (id: string) => void;
  onUpdate: (id: string, patch: KeyframePatch) => void;
  /** The track was clicked at `frame`. Not set: the track does not take clicks. */
  onSeek?: (frame: number) => void;
  /**
   * A marker was pressed or is moving, at `frameIndex`. `null` when the pointer is released.
   * It shows the frame the marker has now, before the move is saved.
   */
  onDragFrame?: (drag: DraggedFrame | null) => void;
  /** Files were dropped on the track. `frame` is the frame under the pointer. Not set: no drop. */
  onDropFiles?: (event: DragEvent<HTMLElement>, frame: number) => void;
  /** Set: each marker has a button that removes its still, and the Delete key removes it too. */
  onRemove?: (id: string) => void;
}) {
  const railRef = useRef<HTMLDivElement>(null);
  const { drag, frameAt, startDrag } = useKeyframeMarkerDrag({
    railRef,
    keyframes,
    lastFrame,
    disabled,
    onSelect,
    onUpdate,
    onDragFrame,
  });
  // Where files would land while they are dragged over the track.
  const [dropFrame, setDropFrame] = useState<number | null>(null);
  const shown = withDraggedFrame(keyframes, drag);
  // The labels follow time, like the inspector. A drag in progress does not renumber them.
  const labels = new Map(byFrame(keyframes).map((keyframe, index) => [keyframe.id, index + 1]));
  const ticks = rulerTicks(lastFrame, fps);
  // The line is at the playhead, or under the pointer while files are dragged over the track.
  const lineFrame = dropFrame ?? playhead;

  const handleKey = (event: KeyboardEvent<HTMLDivElement>, keyframe: FeatureKeyframe) => {
    if (disabled) return;
    // The same removal as the button on the marker, for the keyboard.
    if (onRemove && (event.key === "Delete" || event.key === "Backspace")) {
      event.preventDefault();
      onRemove(keyframe.id);
      return;
    }
    if (!KEYFRAME_EDIT_KEYS.includes(event.key)) return;
    event.preventDefault();
    const patch = keyframePatchForKey(keyframes, keyframe, event.key, lastFrame);
    if (patch) onUpdate(keyframe.id, patch);
  };

  return (
    <div
      className={clsx(styles.strip, disabled && styles.disabled)}
      style={
        {
          "--keyframe-thumb": `${markerSize}px`,
          ...(railHeight === undefined ? {} : { "--keyframe-rail-height": `${railHeight}px` }),
        } as CSSProperties
      }
    >
      <div
        className={styles.track}
        onDragOver={(event) => {
          if (onDropFiles && event.dataTransfer.types.includes("Files")) {
            setDropFrame(frameAt(event.clientX));
          }
        }}
        onDragLeave={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDropFrame(null);
        }}
        onDrop={(event) => {
          if (!onDropFiles) return;
          setDropFrame(null);
          const frame = frameAt(event.clientX);
          if (frame === null) return;
          // The field must not also take this drop as a drop anywhere else.
          event.stopPropagation();
          onDropFiles(event, frame);
        }}
      >
        <div
          ref={railRef}
          className={clsx(styles.rail, onSeek && styles.seekable)}
          onPointerDown={(event) => {
            if (!onSeek || disabled || event.target !== event.currentTarget) return;
            const frame = frameAt(event.clientX);
            if (frame !== null) onSeek(frame);
          }}
        >
          {/* It also shows where dropped files would land, with or without a playhead. */}
          {lineFrame !== undefined ? (
            <span
              className={styles.playhead}
              style={{ left: `${lastFrame > 0 ? (lineFrame / lastFrame) * 100 : 0}%` }}
              aria-hidden
            />
          ) : null}
          {shown.map((keyframe) => (
            <KeyframeMarker
              key={keyframe.id}
              keyframe={keyframe}
              order={labels.get(keyframe.id) ?? 0}
              lastFrame={lastFrame}
              fps={fps}
              label={`Keyframe ${labels.get(keyframe.id)}`}
              isSelected={keyframe.id === selectedId}
              isDragging={drag?.id === keyframe.id}
              onPointerDown={(event) => startDrag(event, keyframe.id)}
              onKeyDown={(event) => handleKey(event, keyframe)}
              // Tab reaches a marker without a pointer. The inspector must then show that still.
              onFocus={() => onSelect(keyframe.id)}
              onRemove={onRemove ? () => onRemove(keyframe.id) : undefined}
            />
          ))}
        </div>
        {/* The ruler shares the width of the rail, so a tick sits under the markers it matches. */}
        <div className={styles.ruler} aria-hidden>
          {ticks.map((tick) => (
            <span
              key={tick.frame}
              className={clsx(styles.tick, tick.label !== null && styles.tickMain)}
              style={{ left: `${(tick.frame / lastFrame) * 100}%` }}
            >
              {tick.label !== null ? <span className={styles.tickLabel}>{tick.label}</span> : null}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
