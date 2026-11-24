import type { CSSProperties } from "react";

import { aspectRatioToNumber } from "@ds/lib/infinityCommon";
import { closestVideoAspectRatio } from "@/lib/closest-video-aspect-ratio";
import { VIDEO_GENERATION_ASPECT_RATIOS } from "@/lib/video-generation-model-specs";

/**
 * The width / height of the frame the pipeline renders for an image. The pipeline
 * center-crops the image to the closest supported ratio, so the editor shows that crop.
 */
export function motionTrackFrameAspect(width: number, height: number): number {
  const ratio = closestVideoAspectRatio(width, height, VIDEO_GENERATION_ASPECT_RATIOS) ?? "16:9";
  return aspectRatioToNumber(ratio);
}

/** Shared max height for modal draw / pace previews. */
export const MOTION_PATH_MODAL_PREVIEW_MAX_HEIGHT = "min(60vh, 560px)";

/** Tighter cap for the form/node ref tile. */
export const MOTION_PATH_FORM_PREVIEW_MAX_HEIGHT = "min(40vh, 320px)";

/**
 * The least size of the pace step, so the modal keeps the height of the draw step.
 * Landscape takes the height of a full-width draw frame. Portrait takes the same height cap.
 * Content that is taller still grows the box.
 */
export function getMotionPathStageMinStyle(imageAspect: number, maxHeight: string): CSSProperties {
  return imageAspect < 1 ? { minHeight: maxHeight } : { aspectRatio: String(imageAspect) };
}

/**
 * Size a preview frame to the image's natural aspect without stretching.
 * Portrait is height-capped; landscape fills available width.
 */
export function getMotionPathPreviewFrameStyle(
  imageAspect: number,
  maxHeight: string,
): CSSProperties {
  if (imageAspect < 1) {
    return {
      aspectRatio: String(imageAspect),
      height: maxHeight,
      width: "auto",
      maxWidth: "100%",
    };
  }
  return {
    aspectRatio: String(imageAspect),
    width: "100%",
    height: "auto",
  };
}
