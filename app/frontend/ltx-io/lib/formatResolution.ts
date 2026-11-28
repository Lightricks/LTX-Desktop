import {
  namedResolutionDisplayName,
  namedResolutionTier,
} from "../../lib/video-resolution.ts";

export type VideoSize = { width: number; height: number };

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
