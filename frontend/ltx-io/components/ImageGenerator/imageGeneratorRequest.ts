import type { components } from "@/generated/backend-openapi";
import type { PromptProvenance } from "@/lib/prompt-provenance";

import {
  IMAGE_RESOLUTIONS,
  IMAGE_SHORT_SIDE_BY_RESOLUTION,
  type ImageResolution,
} from "../../../lib/genspace-image-resolution.ts";
import type { ImageProvider } from "./imageProviderAvailability";

export type CreateTextToImageRequest = components["schemas"]["CreateTextToImageRequest"];

/** The picker order. It follows the feature forms: 16:9 and 9:16 first, then 4:3, then 1:1. */
export const IMAGE_ASPECTS = ["16:9", "9:16", "4:3", "3:4", "1:1"] as const;
export type ImageAspect = (typeof IMAGE_ASPECTS)[number];

/**
 * The image shape closest to a video aspect ratio, or `undefined` for `auto` and others.
 * The video forms offer 21:9, 3:2 and 4:5 too, which the image picker does not, so these
 * map to the nearest shape by ratio, so a 4:5 video does not open a 16:9 image.
 */
export function imageAspectForVideo(videoAspect: string): ImageAspect | undefined {
  const [w, h] = videoAspect.split(":").map(Number);
  if (!w || !h) return undefined;
  const distance = (aspect: ImageAspect) => Math.abs(Math.log(aspectRatioNumber(aspect) / (w / h)));
  return IMAGE_ASPECTS.reduce((best, aspect) => (distance(aspect) < distance(best) ? aspect : best));
}

/** Width divided by height, such as 1.78 for 16:9. */
export function aspectRatioNumber(aspect: ImageAspect): number {
  const [w, h] = aspect.split(":").map(Number) as [number, number];
  return w / h;
}

/** The longest side the backend accepts (`IMAGE_MAX_SIDE` in `backend/api_types.py`). */
export const IMAGE_MAX_SIDE = 2048;

/**
 * The resolution is the short side, as in GenSpace. The backend rounds each side down to a
 * multiple of 16.
 */
export function imageSizeForAspect(
  aspect: ImageAspect,
  resolution: ImageResolution,
): { width: number; height: number } {
  const shortSide = IMAGE_SHORT_SIDE_BY_RESOLUTION[resolution];
  const [w, h] = aspect.split(":").map(Number) as [number, number];
  return w >= h
    ? { width: Math.round((shortSide * w) / h), height: shortSide }
    : { width: shortSide, height: Math.round((shortSide * h) / w) };
}

/** Whether the shape at this resolution stays within the backend limit. */
export function imageResolutionFits(aspect: ImageAspect, resolution: ImageResolution): boolean {
  const { width, height } = imageSizeForAspect(aspect, resolution);
  return Math.max(width, height) <= IMAGE_MAX_SIDE;
}

/** The chosen resolution, or the largest smaller one that fits the shape. */
export function fittingImageResolution(
  aspect: ImageAspect,
  resolution: ImageResolution,
): ImageResolution {
  const fitting = IMAGE_RESOLUTIONS.filter(
    (candidate) =>
      IMAGE_SHORT_SIDE_BY_RESOLUTION[candidate] <= IMAGE_SHORT_SIDE_BY_RESOLUTION[resolution] &&
      imageResolutionFits(aspect, candidate),
  );
  return fitting[fitting.length - 1] ?? IMAGE_RESOLUTIONS[0]!;
}

export type ImageGeneratorSubmission = {
  prompt: string;
  provenance: PromptProvenance;
  size: { width: number; height: number };
  provider: ImageProvider;
};

/** No seed: each request draws its own, which also varies the backend rewrite. */
export function buildTextToImageRequest(input: ImageGeneratorSubmission): CreateTextToImageRequest {
  return {
    contract_version: 1,
    params: {
      prompt: input.prompt,
      promptProvenance: input.provenance,
      provider: input.provider,
      ...input.size,
    },
  };
}
