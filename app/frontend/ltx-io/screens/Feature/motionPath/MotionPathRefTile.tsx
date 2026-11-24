import { type MotionPathPoint, motionPathDotColor } from "../../../lib/motionPathSampler";
import { useCallback, useEffect, useRef } from "react";

import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { Tooltip } from "@ds/Tooltip/Tooltip";
import EditIcon from "@ds/assets/Icons/Edit.svg?react";
import MediaReplaceIcon from "@ds/assets/Icons/MediaReplace.svg?react";
import PhotoIcon from "@ds/assets/Icons/Photo.svg?react";

import styles from "./MotionPathRefTile.module.scss";
import {
  drawMotionPathDots,
  drawMotionPathStroke,
  resizeCanvasToContainer,
  sampleEditorMotionPathFrames,
} from "./motionPathCanvasUtils";
import {
  MOTION_PATH_FORM_PREVIEW_MAX_HEIGHT,
  getMotionPathPreviewFrameStyle,
} from "./motionPathPreviewFrameStyle";

const PATH_SAMPLE_STEPS = 24;

export type MotionPathRefTileProps = {
  /** The points of each track. */
  tracks: MotionPathPoint[][];
  /** The reference image, or null while the tile is empty. */
  imageUrl: string | null;
  /** The frame the pipeline renders for the image, or null until the image loads. */
  imageAspect: number | null;
  onEdit: () => void;
  /** Empty-state click — opens the reference-image picker (Retake-style CTA). */
  onUpload?: () => void;
  /** Replace reference image — shown next to Edit when set. */
  onReplace?: () => void;
  disabled?: boolean;
};

function drawTrackOverlay(
  ctx: CanvasRenderingContext2D,
  points: MotionPathPoint[],
  width: number,
  height: number,
  dotColor: string,
): void {
  if (points.length === 0) return;

  if (points.length >= 2) {
    const allFrames = sampleEditorMotionPathFrames(
      points,
      width,
      height,
      PATH_SAMPLE_STEPS,
    );

    for (let segmentIndex = 0; segmentIndex < points.length - 1; segmentIndex++) {
      const startFrame = segmentIndex * PATH_SAMPLE_STEPS;
      const endFrame = startFrame + PATH_SAMPLE_STEPS;
      const segmentFrames = allFrames.slice(startFrame, endFrame + 1);

      drawMotionPathStroke(ctx, segmentFrames, {
        strokeStyle: "rgba(255, 255, 255, 0.7)",
        lineWidth: 2,
      });
    }
  }

  drawMotionPathDots(ctx, points, {
    width,
    height,
    dotColor,
  });
}

function drawPathOverlay(
  ctx: CanvasRenderingContext2D,
  tracks: MotionPathPoint[][],
  width: number,
  height: number,
): void {
  ctx.clearRect(0, 0, width, height);
  if (width <= 0 || height <= 0) return;
  tracks.forEach((points, index) =>
    drawTrackOverlay(ctx, points, width, height, motionPathDotColor(index)),
  );
}

export function MotionPathRefTile({
  tracks,
  imageUrl,
  imageAspect,
  onEdit,
  onUpload,
  onReplace,
  disabled = false,
}: MotionPathRefTileProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewRef = useRef<HTMLButtonElement>(null);

  const isEmpty = !imageUrl;
  // Empty tile is still interactive when upload is available (mirrors Retake).
  const isInteractionDisabled = disabled || (isEmpty && !onUpload);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const preview = previewRef.current;
    if (!canvas || !preview) return;

    const { width, height } = resizeCanvasToContainer(canvas, preview);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    drawPathOverlay(ctx, tracks, width, height);
  }, [tracks]);

  useEffect(() => {
    draw();
  }, [draw, imageAspect]);

  useEffect(() => {
    const preview = previewRef.current;
    if (!preview) return;
    const observer = new ResizeObserver(() => draw());
    observer.observe(preview);
    return () => observer.disconnect();
  }, [draw]);

  const previewStyle =
    imageAspect !== null
      ? getMotionPathPreviewFrameStyle(imageAspect, MOTION_PATH_FORM_PREVIEW_MAX_HEIGHT)
      : undefined;

  return (
    <div
      className={styles.card}
      data-testid="motion-path-ref-tile"
      data-disabled={isInteractionDisabled || undefined}
      data-empty={isEmpty || undefined}
    >
      <div className={styles.previewShell}>
        <div
          className={styles.previewFrame}
          style={previewStyle}
          data-sized={imageAspect !== null || undefined}
        >
          <button
            ref={previewRef}
            type="button"
            className={styles.preview}
            data-testid="motion-path-ref-preview"
            disabled={isInteractionDisabled}
            aria-label={isEmpty ? "Upload reference image" : "Edit motion path"}
            onClick={isEmpty ? onUpload : onEdit}
          >
            {imageUrl ? (
              <>
                <img
                  className={styles.referenceImage}
                  src={imageUrl}
                  alt=""
                  draggable={false}
                />
                <canvas ref={canvasRef} className={styles.pathOverlay} aria-hidden />
              </>
            ) : (
              <div className={styles.emptyState}>
                <PhotoIcon className={styles.emptyIcon} aria-hidden />
                <Text as="p" variant="body" size="md" className={styles.emptyTitle}>
                  Add a start frame
                </Text>
              </div>
            )}
          </button>
          {!isEmpty ? (
            <div className={styles.overlayActions}>
              {onReplace ? (
                <Tooltip content="Replace Start Frame">
                  <span>
                    <Button
                      appearance="overlay"
                      hierarchy="secondary"
                      size="md"
                      isIconOnly
                      leftIcon={<MediaReplaceIcon />}
                      aria-label="Replace Start Frame"
                      disabled={disabled}
                      onClick={(event) => {
                        event.stopPropagation();
                        onReplace();
                      }}
                    />
                  </span>
                </Tooltip>
              ) : null}
              <Tooltip content="Edit motion path">
                <span>
                  <Button
                    appearance="overlay"
                    hierarchy="secondary"
                    size="md"
                    isIconOnly
                    leftIcon={<EditIcon />}
                    aria-label="Edit motion path"
                    disabled={disabled}
                    onClick={(event) => {
                      event.stopPropagation();
                      onEdit();
                    }}
                  />
                </span>
              </Tooltip>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
