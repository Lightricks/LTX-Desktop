import {
  getHomeFeature,
  type HomeFeatureId,
} from "../../../lib/home-features.ts";

export const QUICK_SEARCH_PLACEHOLDER = "Search tools";

/** Rotating hints — catalog titles so typed text matches search results. */
const PLACEHOLDER_FEATURE_IDS = [
  "text-to-video",
  "image-to-video",
  "audio-to-video",
  "extend",
  "retake",
  "cozy-felt",
  "dolly-in",
  "cinemagraph",
] as const satisfies readonly HomeFeatureId[];

export const QUICK_SEARCH_TOOL_PLACEHOLDERS: string[] = PLACEHOLDER_FEATURE_IDS.map(
  (id) => getHomeFeature(id).title,
);
