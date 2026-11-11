import type { ExploreListedAsset } from "../../../lib/explore-contract.ts";
import {
  IMAGE_RESOLUTIONS,
  IMAGE_SHORT_SIDE_BY_RESOLUTION,
  IMAGE_SIZE_GRID,
} from "../../../lib/genspace-image-resolution.ts";

import { formatDurationBadge } from "../../lib/generationPresentation.ts";
import { formatResolution, type VideoSize } from "../../lib/formatResolution.ts";

export { formatResolution };

const KNOWN_ASPECT_RATIOS: ReadonlyArray<{ label: string; value: number }> = [
  { label: "21:9", value: 21 / 9 },
  { label: "16:9", value: 16 / 9 },
  { label: "3:2", value: 3 / 2 },
  { label: "4:3", value: 4 / 3 },
  { label: "5:4", value: 5 / 4 },
  { label: "1:1", value: 1 },
  { label: "4:5", value: 4 / 5 },
  { label: "3:4", value: 3 / 4 },
  { label: "2:3", value: 2 / 3 },
  { label: "9:16", value: 9 / 16 },
  { label: "9:21", value: 9 / 21 },
];

/** Allowed relative gap between the real ratio and a known one, for example 1280x704 vs 16:9. */
const ASPECT_RATIO_TOLERANCE = 0.03;

const SECONDS_PER_MINUTE = 60;

export function formatClipDuration(durationMs: number): string | null {
  if (!Number.isFinite(durationMs) || durationMs <= 0) return null;
  const seconds = durationMs / 1000;
  if (seconds < SECONDS_PER_MINUTE) return formatDurationBadge(seconds);
  const total = Math.round(seconds);
  const minutes = Math.floor(total / SECONDS_PER_MINUTE);
  const rest = total % SECONDS_PER_MINUTE;
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

/** Returns the nearest known ratio label, or null when the clip is an unusual shape. */
export function formatAspectRatio({ width, height }: VideoSize): string | null {
  if (!(width > 0) || !(height > 0)) return null;
  const ratio = width / height;
  let best: { label: string; gap: number } | null = null;
  for (const known of KNOWN_ASPECT_RATIOS) {
    const gap = Math.abs(ratio - known.value) / known.value;
    if (best === null || gap < best.gap) best = { label: known.label, gap };
  }
  return best !== null && best.gap <= ASPECT_RATIO_TOLERANCE ? best.label : null;
}

/** Ordered parts for the tile badge: duration, aspect ratio, resolution. */
export function formatVideoMeta(metadata: ExploreListedAsset["metadata"]): string[] {
  if (metadata.mediaType !== "video") return [];
  const { durationMs, width, height } = metadata.metadata;
  const size = { width, height };
  return [
    formatClipDuration(durationMs),
    formatAspectRatio(size),
    formatResolution(size),
  ].filter((part): part is string => part !== null);
}

/**
 * The short side of an image, such as "1024p". It is not snapped to a video tier, so a
 * 1024x1024 image does not read "1080p". An image that sits one grid step below an image tier
 * reads as that tier: the backend rounds 1080 down to 1072, and the picker said "1080p".
 */
export function formatImageResolution({ width, height }: VideoSize): string | null {
  if (!(width > 0) || !(height > 0)) return null;
  const shortSide = Math.min(width, height);
  const tier = IMAGE_RESOLUTIONS.find((resolution) => {
    const gap = IMAGE_SHORT_SIDE_BY_RESOLUTION[resolution] - shortSide;
    return gap >= 0 && gap < IMAGE_SIZE_GRID;
  });
  return tier ?? `${shortSide}p`;
}

/** Ordered parts for the tile badge of an image: aspect ratio, resolution. */
export function formatImageMeta(metadata: ExploreListedAsset["metadata"]): string[] {
  if (metadata.mediaType !== "image") return [];
  const size = { width: metadata.metadata.width, height: metadata.metadata.height };
  return [formatAspectRatio(size), formatImageResolution(size)].filter(
    (part): part is string => part !== null,
  );
}

/** Badge parts for the asset. Audio has no badge. */
export function formatAssetMeta(metadata: ExploreListedAsset["metadata"]): string[] {
  switch (metadata.mediaType) {
    case "video":
      return formatVideoMeta(metadata);
    case "image":
      return formatImageMeta(metadata);
    case "audio":
      return [];
    default: {
      const _exhaustive: never = metadata;
      return _exhaustive;
    }
  }
}
