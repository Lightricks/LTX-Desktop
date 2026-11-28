import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import PhotosIcon from "@ds/assets/Icons/Photos.svg?react";
import { clsx } from "clsx";
import { useEffect, useState } from "react";

import { useIsPhoneViewport } from "@/ltx-io/hooks/useIsPhoneViewport";
import { AssetsLightbox } from "@/ltx-io/screens/Assets/AssetsLightbox";

import { lastFrameFromDuration } from "../../../../../lib/keyframe-timeline.ts";
import formStyles from "../../FeatureFormView.module.scss";
import type { AssetRef, FeatureKeyframe, ValidationIssue } from "../../types";
import dropStyles from "../ImageDropzone.module.scss";
import { IMAGE_ACCEPT } from "../imageAssetInput";

import { KeyframeRow } from "./KeyframeRow";
import { KeyframeTimelineStrip } from "./KeyframeTimelineStrip";
import { KeyframesTimelineModal } from "./KeyframesTimelineModal";
import styles from "./KeyframesField.module.scss";
import {
  type KeyframePatch,
  applyKeyframeLayout,
  byFrame,
  markerSizeForCount,
  removeKeyframe,
  spreadKeyframes,
  swapKeyframeFrames,
  updateKeyframe,
} from "./keyframesModel";
import { SortableKeyframeList } from "./SortableKeyframeList";
import { useKeyframeLightboxAssets } from "./useKeyframeLightboxAssets";
import { useKeyframeEdit } from "./useKeyframeEdit";
import { useKeyframePreview } from "./useKeyframePreview";
import { useKeyframesIngest } from "./useKeyframesIngest";

/**
 * Stills at chosen times of the clip, each with its own strength. One drop zone while the list
 * is empty. Then a timeline and a list with one row for each still, in time order. A click on a
 * thumbnail opens the still in a big preview that steps through all the stills.
 * The clip length and the frame rate come from other fields. This field never changes them.
 */
export function KeyframesField({
  field,
  value,
  maxCount,
  duration,
  fps,
  issue,
  onChange,
}: {
  field: { id: string; label: string };
  value: FeatureKeyframe[];
  maxCount: number;
  /** The clip length in seconds. Not a number while no length is chosen. */
  duration: number;
  fps: number;
  issue?: ValidationIssue;
  onChange: (value: FeatureKeyframe[]) => void;
}) {
  const hasTimeline = Number.isFinite(duration) && duration > 0 && fps > 0;
  const lastFrame = hasTimeline ? lastFrameFromDuration(duration, fps) : 0;
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // The big timeline needs room. A phone has no button for it, and a window that turns into
  // a phone size closes it.
  const [timelineOpen, setTimelineOpen] = useState(false);
  const isPhone = useIsPhoneViewport();
  const timelineAvailable = hasTimeline && !isPhone && value.length > 0;
  // A modal that cannot show must not open again by itself when it can.
  useEffect(() => {
    if (!timelineAvailable) setTimelineOpen(false);
  }, [timelineAvailable]);

  const edit = useKeyframeEdit(value, onChange);
  const { fileInputRef, addError, clearAddError, isDragOver, dropProps, dropAt } =
    useKeyframesIngest({
      label: field.label,
      maxCount,
      count: value.length,
      lastFrame,
      preferredFrame: 0,
      edit,
      onAdded: (added) => setSelectedId(added.id),
    });

  const ordered = byFrame(value);
  const lightbox = useKeyframeLightboxAssets(
    ordered.map((keyframe) => keyframe.asset.assetId),
  );
  const preview = useKeyframePreview(
    ordered.map((keyframe) => keyframe.id),
    lightbox,
  );
  const message = addError ?? issue?.message;
  const canAdd = hasTimeline && value.length < maxCount;

  const change = (update: (current: FeatureKeyframe[]) => FeatureKeyframe[]) => {
    // The old add message is about a list that no longer exists.
    clearAddError();
    edit(update);
  };

  return (
    <div className={formStyles.field}>
      <div className={styles.header}>
        <Text as="span" variant="body" size="md" className={formStyles.fieldLabel}>
          {field.label}
        </Text>
        {value.length > 0 ? (
          <div className={styles.headerEnd}>
            <Text as="span" variant="body" size="sm" className={styles.count}>
              {`${value.length}/${maxCount}`}
            </Text>
            <Button
              appearance="neutral"
              hierarchy="secondary"
              size="sm"
              label="Add"
              disabled={!canAdd}
              onClick={dropProps.onClick}
            />
            <Button
              appearance="neutral"
              hierarchy="secondary"
              size="sm"
              label="Spread evenly"
              disabled={!hasTimeline || value.length < 2}
              onClick={() => change((current) => spreadKeyframes(current, lastFrame))}
            />
            {hasTimeline && !isPhone ? (
              <Button
                appearance="neutral"
                hierarchy="secondary"
                size="sm"
                label="Expand"
                onClick={() => setTimelineOpen(true)}
              />
            ) : null}
            <Button
              appearance="neutral"
              hierarchy="secondary"
              size="sm"
              label="Clear all"
              onClick={() => {
                setSelectedId(null);
                change(() => []);
              }}
            />
          </div>
        ) : null}
      </div>
      <input
        ref={fileInputRef}
        className={dropStyles.hiddenInput}
        type="file"
        accept={IMAGE_ACCEPT}
        multiple
      />
      {value.length === 0 ? (
        <button
          type="button"
          className={clsx(dropStyles.dropzone, isDragOver && dropStyles.dragOver)}
          disabled={!hasTimeline}
          {...dropProps}
        >
          <span className={dropStyles.empty}>
            <PhotosIcon className={dropStyles.emptyIcon} aria-hidden />
            <Text as="span" variant="label" size="md">
              Add keyframes
            </Text>
            <Text as="span" variant="body" size="md" className={styles.subtitle}>
              {hasTimeline
                ? `Place up to ${maxCount} images at chosen times of the video`
                : "Choose a video length first"}
            </Text>
          </span>
        </button>
      ) : (
        <div
          className={clsx(styles.editor, isDragOver && dropStyles.dragOver)}
          onDragEnter={dropProps.onDragEnter}
          onDragOver={dropProps.onDragOver}
          onDragLeave={dropProps.onDragLeave}
          onDrop={dropProps.onDrop}
        >
          <KeyframeTimelineStrip
            keyframes={value}
            lastFrame={lastFrame}
            fps={fps}
            selectedId={selectedId}
            markerSize={markerSizeForCount(value.length)}
            disabled={!hasTimeline}
            onSelect={setSelectedId}
            onUpdate={(id: string, patch: KeyframePatch) =>
              change((current) => updateKeyframe(current, id, patch))
            }
            onDropFiles={dropAt}
          />
          {hasTimeline ? (
            <SortableKeyframeList
              orderedIds={ordered.map((keyframe) => keyframe.id)}
              onSwap={(activeId, overId) =>
                change((current) => swapKeyframeFrames(current, activeId, overId))
              }
            >
              <div className={styles.list}>
                {ordered.map((keyframe, index) => (
                  <KeyframeRow
                    key={keyframe.id}
                    keyframe={keyframe}
                    keyframes={value}
                    index={index}
                    fps={fps}
                    lastFrame={lastFrame}
                    label={field.label}
                    isSelected={keyframe.id === selectedId}
                    canReorder={ordered.length > 1}
                    onSelect={() => setSelectedId(keyframe.id)}
                    // A still whose image did not load has nothing to show.
                    onPreview={
                      lightbox.positions[index] === null
                        ? null
                        : () => preview.open(keyframe.id)
                    }
                    onUpdate={(patch) =>
                      change((current) => updateKeyframe(current, keyframe.id, patch))
                    }
                    // The upload can end after this row is gone. A removed still stays removed.
                    onReplace={(asset: AssetRef) =>
                      change((current) => updateKeyframe(current, keyframe.id, { asset }))
                    }
                    onRemove={() => {
                      if (selectedId === keyframe.id) setSelectedId(null);
                      change((current) => removeKeyframe(current, keyframe.id));
                    }}
                  />
                ))}
              </div>
            </SortableKeyframeList>
          ) : null}
        </div>
      )}
      {timelineOpen && timelineAvailable ? (
        <KeyframesTimelineModal
          keyframes={value}
          lastFrame={lastFrame}
          fps={fps}
          selectedId={selectedId}
          label={field.label}
          maxCount={maxCount}
          // The list can change while the modal is open. The modal edits a copy, so its stills
          // go on the list as it is when the user presses Done.
          onApply={(edited, base) => {
            change((current) =>
              applyKeyframeLayout(current, base, edited, { lastFrame, maxCount }),
            );
            setTimelineOpen(false);
          }}
          onClose={() => setTimelineOpen(false)}
        />
      ) : null}
      {preview.lightboxProps ? (
        // The still is an input. The preview does not offer its file in the file manager.
        <AssetsLightbox {...preview.lightboxProps} allowReveal={false} />
      ) : null}
      {message ? <span className={formStyles.fieldError}>{message}</span> : null}
    </div>
  );
}
