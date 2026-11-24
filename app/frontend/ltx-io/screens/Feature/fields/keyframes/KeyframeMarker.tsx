import RemoveIcon from "@ds/assets/Icons/Remove.svg?react";
import { clsx } from "clsx";
import type { KeyboardEvent, PointerEvent } from "react";

import { formatKeyframeStrength } from "../../../../../lib/keyframe-strength.ts";
import { formatKeyframeSeconds } from "../../../../../lib/keyframe-timeline.ts";
import type { FeatureKeyframe } from "../../types";
import styles from "./KeyframeTimelineStrip.module.scss";
import { useAssetThumbUrl } from "./useAssetThumbUrl";

/** One still on the rail: its image, its order, and the strength when it is selected. */
export function KeyframeMarker({
  keyframe,
  order,
  lastFrame,
  fps,
  label,
  isSelected,
  isDragging,
  onPointerDown,
  onKeyDown,
  onFocus,
  onRemove,
}: {
  keyframe: FeatureKeyframe;
  /** The place of the still in time order. The list row of the still shows the same number. */
  order: number;
  lastFrame: number;
  fps: number;
  label: string;
  isSelected: boolean;
  isDragging: boolean;
  onPointerDown: (event: PointerEvent<HTMLDivElement>) => void;
  onKeyDown: (event: KeyboardEvent<HTMLDivElement>) => void;
  onFocus: () => void;
  /** Set: the marker has a button that removes its still, for the pointer. Not set: no button. */
  onRemove?: () => void;
}) {
  const thumbUrl = useAssetThumbUrl(keyframe.asset.assetId);
  const position = keyframe.frameIndex;
  return (
    <div
      role="slider"
      tabIndex={0}
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={lastFrame}
      aria-valuenow={position}
      aria-valuetext={`${formatKeyframeSeconds(position, fps)}, strength ${formatKeyframeStrength(keyframe.strength)}`}
      // The frame number is for the user who needs it. It is not on the marker.
      title={`${formatKeyframeSeconds(position, fps)} · frame ${position}`}
      data-keyframe-id={keyframe.id}
      // A modal that holds the strip puts the focus on the selected marker when it opens.
      data-autofocus={isSelected ? "" : undefined}
      className={clsx(
        styles.marker,
        isSelected && styles.selected,
        isDragging && styles.dragging,
      )}
      style={{ left: `${lastFrame > 0 ? (position / lastFrame) * 100 : 0}%` }}
      onPointerDown={onPointerDown}
      onKeyDown={onKeyDown}
      onFocus={onFocus}
    >
      {thumbUrl ? <img className={styles.markerImage} src={thumbUrl} alt="" draggable={false} /> : null}
      <span className={styles.markerOrder}>{order}</span>
      {onRemove ? (
        // A slider has no interactive children for a screen reader, and this button is for the
        // pointer. The Delete key on the marker removes the still for the keyboard.
        <button
          type="button"
          className={styles.markerRemove}
          tabIndex={-1}
          aria-hidden="true"
          title="Remove"
          // The press must not start a drag of the marker.
          onPointerDown={(event) => event.stopPropagation()}
          onClick={onRemove}
        >
          <RemoveIcon aria-hidden />
        </button>
      ) : null}
      {isSelected ? (
        <span className={styles.markerStrength}>{formatKeyframeStrength(keyframe.strength)}</span>
      ) : null}
    </div>
  );
}
