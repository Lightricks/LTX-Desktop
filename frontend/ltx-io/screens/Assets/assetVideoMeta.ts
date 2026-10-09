import type { ExploreListedAsset } from "../../../lib/explore-contract.ts";

import {
  namedResolutionDisplayName,
  namedResolutionTier,
} from "../../../lib/video-resolution.ts";
import { formatDurationBadge } from "../../lib/generationPresentation.ts";

type VideoSize = { width: number; height: number };

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

/**
 * The generation resolution ids, by short side (270p ... 2160p). `namedResolutionTier`
 * stops at 540 because the editor depends on it, so the two small tiers are added here.
 */
const LOW_RESOLUTION_TIERS = [270, 360] as const;

/** Allowed relative gap between the short side and its tier, for example 704 vs 720. */
const RESOLUTION_TOLERANCE = 0.1;

function nearestResolutionTier(shortSide: number): number {
  return LOW_RESOLUTION_TIERS.reduce<number>(
    (best, low) => (Math.abs(low - shortSide) < Math.abs(best - shortSide) ? low : best),
    namedResolutionTier(shortSide),
  );
}

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

/**
 * Snaps the short side to the nearest generation tier, so 1920x1080 and 1080x1920 both
 * read "1080p", 1280x704 reads "720p", 1024x576 reads "540p" and 454x256 reads "270p".
 * A clip far from the nearest tier keeps its own short side, so 854x480 reads "480p" and
 * 320x240 reads "240p". The Retake "(Original)" labels follow the same rule.
 */
export function formatResolution({ width, height }: VideoSize): string | null {
  if (!(width > 0) || !(height > 0)) return null;
  const shortSide = Math.min(width, height);
  const tier = nearestResolutionTier(shortSide);
  return Math.abs(shortSide - tier) / tier <= RESOLUTION_TOLERANCE
    ? namedResolutionDisplayName(tier)
    : `${shortSide}p`;
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
