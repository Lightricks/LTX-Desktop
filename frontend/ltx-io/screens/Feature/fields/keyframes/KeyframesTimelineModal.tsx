import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { clsx } from "clsx";
import { useEffect, useRef, useState } from "react";

import { Modal } from "@/ltx-io/components/shared/Modal/Modal";

import {
  type DraggedFrame,
  formatKeyframeSeconds,
  previewKeyframeForPlayhead,
} from "../../../../../lib/keyframe-timeline.ts";
import formStyles from "../../FeatureFormView.module.scss";
import type { FeatureKeyframe } from "../../types";
import dropStyles from "../ImageDropzone.module.scss";
import { IMAGE_ACCEPT } from "../imageAssetInput";

import { KeyframePreviewImage } from "./KeyframePreviewImage";
import { KeyframeTimelineStrip } from "./KeyframeTimelineStrip";
import styles from "./KeyframesTimelineModal.module.scss";
import {
  type KeyframePatch,
  byFrame,
  removeKeyframe,
  spreadKeyframes,
  updateKeyframe,
} from "./keyframesModel";
import { useKeyframeEdit } from "./useKeyframeEdit";
import { useKeyframesIngest } from "./useKeyframesIngest";

/** The side of a marker and the height of the rail. Ten of these fit the modal width. */
const MARKER_SIZE_PX = 64;
const RAIL_HEIGHT_PX = 72;

/**
 * The timeline on a big surface, for the user who works with many stills. The preview above the
 * timeline shows the image the clip holds at the playhead: the still on it, else the last one
 * before it. A press on a marker moves the playhead to it, and a drag moves the playhead along,
 * so the preview follows the marker. The user can also add stills, at the playhead or where files
 * are dropped on the timeline, and remove one with the button on its marker.
 * The user edits a copy of the stills. Done gives the copy to the field, with the list it started
 * from. Cancel, Escape and a click outside the modal drop the copy, so the field does not change.
 */
export function KeyframesTimelineModal({
  keyframes: initialKeyframes,
  lastFrame,
  fps,
  label,
  maxCount,
  selectedId: initialSelectedId,
  onApply,
  onClose,
}: {
  /** The stills when the modal opens. */
  keyframes: readonly FeatureKeyframe[];
  lastFrame: number;
  fps: number;
  /** The name of the field, for the file chooser. */
  label: string;
  maxCount: number;
  /** The still that is selected when the modal opens. The modal keeps its own selection. */
  selectedId: string | null;
  /** Done. The edited copy of the stills, and the stills the modal started with. */
  onApply: (keyframes: FeatureKeyframe[], base: FeatureKeyframe[]) => void;
  /** Cancel, Escape or a click outside. Nothing was applied. */
  onClose: () => void;
}) {
  const [base] = useState<FeatureKeyframe[]>(() => [...initialKeyframes]);
  const [keyframes, setKeyframes] = useState<FeatureKeyframe[]>(base);
  const edit = useKeyframeEdit(keyframes, setKeyframes);
  // The selection is part of the copy. A Cancel leaves the selection of the field as it was.
  const [selectedId, setSelectedId] = useState(initialSelectedId);
  const [playhead, setPlayhead] = useState(
    () => base.find((keyframe) => keyframe.id === initialSelectedId)?.frameIndex ?? 0,
  );
  // The marker in motion. Its frame is not saved until the pointer is released.
  const [drag, setDrag] = useState<DraggedFrame | null>(null);

  const { fileInputRef, addError, clearAddError, isDragOver, isImporting, dropProps, dropAt } =
    useKeyframesIngest({
      label,
      maxCount,
      count: keyframes.length,
      lastFrame,
      preferredFrame: playhead,
      edit,
      onAdded: (added) => {
        setSelectedId(added.id);
        setPlayhead(added.frameIndex);
      },
    });

  // The marker that gets the focus when its neighbor is removed, so the focus does not fall away.
  const contentRef = useRef<HTMLDivElement>(null);
  const focusAfterRemove = useRef<string | null>(null);
  useEffect(() => {
    const id = focusAfterRemove.current;
    focusAfterRemove.current = null;
    if (id === null) return;
    const markers = contentRef.current?.querySelectorAll<HTMLElement>("[data-keyframe-id]") ?? [];
    Array.from(markers)
      .find((marker) => marker.dataset.keyframeId === id)
      ?.focus();
  }, [keyframes]);

  const frame = drag?.frameIndex ?? playhead;
  const shown = previewKeyframeForPlayhead(keyframes, frame, drag);
  const order = shown ? byFrame(keyframes).findIndex((keyframe) => keyframe.id === shown.id) + 1 : 0;

  return (
    <Modal className={styles.modal} variant="primary" onClose={onClose}>
      <div
        ref={contentRef}
        className={clsx(styles.content, isDragOver && dropStyles.dragOver)}
        data-testid="keyframes-timeline-modal"
        // A file dropped on the modal adds a still. The browser does not open the file.
        onDragEnter={dropProps.onDragEnter}
        onDragOver={dropProps.onDragOver}
        onDragLeave={dropProps.onDragLeave}
        onDrop={dropProps.onDrop}
      >
        <input
          ref={fileInputRef}
          // The modal must not put its focus on this input, which has no box. It skips this element.
          aria-hidden="true"
          className={dropStyles.hiddenInput}
          type="file"
          accept={IMAGE_ACCEPT}
          multiple
        />
        <div className={styles.header}>
          <Text as="h2" variant="heading" size="sm">
            Keyframes
          </Text>
          <div className={styles.actions}>
            <Text as="span" variant="body" size="sm" className={styles.hint}>
              {`${keyframes.length}/${maxCount}`}
            </Text>
            <Button
              appearance="neutral"
              hierarchy="secondary"
              size="sm"
              label="Add"
              disabled={keyframes.length >= maxCount}
              onClick={dropProps.onClick}
            />
            <Button
              appearance="neutral"
              hierarchy="secondary"
              size="sm"
              label="Spread evenly"
              disabled={keyframes.length < 2}
              onClick={() => edit((current) => spreadKeyframes(current, lastFrame))}
            />
          </div>
        </div>
        <div className={styles.preview}>
          {shown ? (
            <KeyframePreviewImage assetId={shown.asset.assetId} className={styles.previewImage} />
          ) : (
            <Text as="span" variant="body" size="md" className={styles.hint}>
              No keyframes. Add images, or drop them here.
            </Text>
          )}
          {shown ? (
            <Text as="span" variant="label" size="sm" className={styles.caption}>
              {`Keyframe ${order} · ${formatKeyframeSeconds(shown.frameIndex, fps)}`}
            </Text>
          ) : null}
        </div>
        <KeyframeTimelineStrip
          keyframes={keyframes}
          lastFrame={lastFrame}
          fps={fps}
          selectedId={selectedId}
          playhead={playhead}
          markerSize={MARKER_SIZE_PX}
          railHeight={RAIL_HEIGHT_PX}
          onSelect={(id) => {
            setSelectedId(id);
            // A marker that gets focus or a press takes the playhead, so the preview shows it.
            const still = keyframes.find((keyframe) => keyframe.id === id);
            if (still) setPlayhead(still.frameIndex);
          }}
          onUpdate={(id: string, patch: KeyframePatch) => {
            edit((current) => updateKeyframe(current, id, patch));
            if (patch.frameIndex !== undefined) setPlayhead(patch.frameIndex);
          }}
          onSeek={setPlayhead}
          onDragFrame={(next) => {
            setDrag(next);
            if (next) setPlayhead(next.frameIndex);
          }}
          // Files dropped on the timeline go near the frame under the pointer.
          onDropFiles={dropAt}
          onRemove={(id) => {
            const ordered = byFrame(keyframes);
            const index = ordered.findIndex((keyframe) => keyframe.id === id);
            focusAfterRemove.current = (ordered[index + 1] ?? ordered[index - 1])?.id ?? null;
            clearAddError();
            edit((current) => removeKeyframe(current, id));
            if (selectedId === id) setSelectedId(null);
          }}
        />
        {addError ? <span className={formStyles.fieldError}>{addError}</span> : null}
        {/* The region is always there, so a screen reader announces the text when it comes. */}
        <div role="status">
          {isImporting ? (
            <Text as="span" variant="body" size="sm" className={styles.hint}>
              Adding images…
            </Text>
          ) : null}
        </div>
        <div className={styles.footer}>
          <Text as="span" variant="body" size="sm" className={styles.hint}>
            Drag a still to move it. Click the track to see the video at that time. Arrow keys
            move the selected still by one frame. Delete removes it. Done saves the changes. Cancel drops
            them.
          </Text>
          <div className={styles.actions}>
            <Button
              appearance="neutral"
              hierarchy="plain"
              size="md"
              label="Cancel"
              onClick={onClose}
            />
            <Button
              appearance="brand"
              hierarchy="primary"
              size="md"
              label="Done"
              // The images that are on their way are not in the copy. Done would lose them.
              disabled={isImporting}
              onClick={() => onApply(keyframes, base)}
            />
          </div>
        </div>
      </div>
    </Modal>
  );
}
