import { useSortable } from "@dnd-kit/sortable";
import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { TextField } from "@ds/TextField/TextField";
import MediaReplaceIcon from "@ds/assets/Icons/MediaReplace.svg?react";
import RemoveIcon from "@ds/assets/Icons/Remove.svg?react";
import { clsx } from "clsx";
import { useEffect, useRef, useState } from "react";

import { applySeconds } from "../../../../../lib/keyframe-controls.ts";
import {
  KEYFRAME_STRENGTH_STEP,
  clampKeyframeStrength,
  formatKeyframeStrength,
} from "../../../../../lib/keyframe-strength.ts";
import { formatKeyframeSeconds } from "../../../../../lib/keyframe-timeline.ts";
import formStyles from "../../FeatureFormView.module.scss";
import type { AssetRef, FeatureKeyframe } from "../../types";
import dropStyles from "../ImageDropzone.module.scss";
import { DragGrip } from "../../../../components/shared/DragGrip/DragGrip";
import { useImageAssetField } from "../useImageAssetField";

import styles from "./KeyframeRow.module.scss";
import type { KeyframePatch } from "./keyframesModel";
import { useAssetThumbUrl } from "./useAssetThumbUrl";

const STRENGTH_PERCENT_STEP = Math.round(KEYFRAME_STRENGTH_STEP * 100);

/**
 * One still in the list: its thumbnail, its time, its strength, and a way to replace or remove it.
 * The grip at the start drags the row onto another row to trade places with it. It must sit in a
 * `SortableKeyframeList`. A click on the thumbnail opens the big preview. The row of the selected still is highlighted,
 * and it scrolls into view when the user selects the still on the timeline.
 */
export function KeyframeRow({
  keyframe,
  keyframes,
  index,
  fps,
  lastFrame,
  label,
  isSelected,
  canReorder,
  onSelect,
  onPreview,
  onUpdate,
  onReplace,
  onRemove,
}: {
  keyframe: FeatureKeyframe;
  /** All the stills, to keep the new time free of the others. */
  keyframes: readonly FeatureKeyframe[];
  index: number;
  fps: number;
  lastFrame: number;
  label: string;
  isSelected: boolean;
  /** False with one still: there is nothing to trade places with. */
  canReorder: boolean;
  onSelect: () => void;
  /** Null when the image is not available, so the thumbnail cannot open a preview. */
  onPreview: (() => void) | null;
  onUpdate: (patch: KeyframePatch) => void;
  onReplace: (asset: AssetRef) => void;
  onRemove: () => void;
}) {
  const thumbUrl = useAssetThumbUrl(keyframe.asset.assetId);
  const slot = useImageAssetField({
    value: keyframe.asset,
    chooserTitle: label,
    onChange: (next) => {
      if (next) onReplace(next);
    },
  });
  // The text the user is typing. It is null when the field shows the saved time.
  const [draft, setDraft] = useState<string | null>(null);
  const seconds = formatKeyframeSeconds(keyframe.frameIndex, fps);
  const rowRef = useRef<HTMLDivElement | null>(null);
  const sortable = useSortable({
    id: keyframe.id,
    disabled: !canReorder,
    animateLayoutChanges: () => false,
  });
  const setRowNode = (node: HTMLDivElement | null) => {
    rowRef.current = node;
    sortable.setNodeRef(node);
  };

  useEffect(() => {
    if (isSelected) rowRef.current?.scrollIntoView?.({ block: "nearest" });
  }, [isSelected]);

  const commitSeconds = () => {
    if (draft === null) return;
    const frameIndex = applySeconds(keyframes, keyframe.id, draft, fps, lastFrame);
    if (frameIndex !== null) onUpdate({ frameIndex });
    setDraft(null);
  };

  return (
    <div
      ref={setRowNode}
      className={clsx(
        styles.row,
        isSelected && styles.selected,
        sortable.isDragging && styles.dragging,
      )}
      style={{
        transform: sortable.transform
          ? `translate3d(${sortable.transform.x}px, ${sortable.transform.y}px, 0)`
          : undefined,
        transition: sortable.transition,
      }}
      data-keyframe-row-id={keyframe.id}
      onPointerDown={onSelect}
      onFocus={onSelect}
    >
      <button
        ref={sortable.setActivatorNodeRef}
        type="button"
        className={styles.dragHandle}
        aria-label={`Reorder keyframe ${index + 1}`}
        title="Drag onto another keyframe to trade places"
        disabled={!canReorder}
        {...sortable.attributes}
        {...sortable.listeners}
      >
        <DragGrip />
      </button>
      <input
        ref={slot.fileInputRef}
        className={dropStyles.hiddenInput}
        type="file"
        accept={slot.accept}
      />
      <button
        type="button"
        className={styles.thumb}
        aria-label={`Preview keyframe ${index + 1}`}
        disabled={onPreview === null}
        onClick={onPreview ?? undefined}
      >
        {thumbUrl ? <img className={styles.thumbImage} src={thumbUrl} alt="" /> : null}
      </button>
      <div className={styles.controls}>
        <div className={styles.line}>
          <Text as="span" variant="body" size="md" className={formStyles.fieldLabel}>
            {`Keyframe ${index + 1}`}
          </Text>
          <div className={styles.actions}>
            <Button
              appearance="neutral"
              hierarchy="secondary"
              size="sm"
              isIconOnly
              leftIcon={<MediaReplaceIcon />}
              aria-label={`Replace image of keyframe ${index + 1}`}
              onClick={slot.openLibrary}
            />
            <Button
              appearance="neutral"
              hierarchy="secondary"
              size="sm"
              isIconOnly
              leftIcon={<RemoveIcon />}
              aria-label={`Remove keyframe ${index + 1}`}
              onClick={onRemove}
            />
          </div>
        </div>
        {/* A form, so that Enter commits the time. The tooltip has the frame the time lands on. */}
        <form
          className={styles.line}
          title={`Frame ${keyframe.frameIndex}`}
          onSubmit={(event) => {
            event.preventDefault();
            commitSeconds();
          }}
        >
          <Text as="span" variant="body" size="md" className={styles.lineLabel}>
            Time
          </Text>
          <TextField
            className={styles.time}
            aria-label={`Keyframe ${index + 1} time in seconds`}
            value={draft ?? seconds}
            onInputChange={setDraft}
            onBlur={commitSeconds}
          />
        </form>
        <div className={styles.line}>
          <Text as="span" variant="body" size="md" className={styles.lineLabel}>
            Strength
          </Text>
          <div className={formStyles.sliderControl}>
            <input
              type="range"
              className={formStyles.slider}
              min={0}
              max={100}
              step={STRENGTH_PERCENT_STEP}
              value={Math.round(clampKeyframeStrength(keyframe.strength) * 100)}
              aria-label={`Keyframe ${index + 1} strength`}
              onChange={(event) =>
                onUpdate({ strength: Number(event.currentTarget.value) / 100 })
              }
            />
            <span className={formStyles.sliderValue}>
              {formatKeyframeStrength(keyframe.strength)}
            </span>
          </div>
        </div>
        {slot.errorMessage ? (
          <span className={formStyles.fieldError}>{slot.errorMessage}</span>
        ) : null}
      </div>
    </div>
  );
}
