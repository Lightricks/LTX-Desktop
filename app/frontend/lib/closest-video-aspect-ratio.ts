import type { VideoGenerationAspectRatio } from "./video-generation-model-specs.ts";

/**
 * The option whose ratio is closest to `width / height`, or null when the height is not positive.
 * The backend twin is `find_closest_aspect_ratio_option`.
 */
export function closestVideoAspectRatio(
  width: number,
  height: number,
  options: readonly VideoGenerationAspectRatio[],
): VideoGenerationAspectRatio | null {
  if (!(height > 0)) return null;
  const target = width / height;
  let closest: VideoGenerationAspectRatio | null = null;
  let closestDiff = Number.POSITIVE_INFINITY;
  for (const option of options) {
    const [rawWidth, rawHeight] = option.split(":");
    const optionWidth = Number(rawWidth);
    const optionHeight = Number(rawHeight);
    if (!(optionWidth > 0) || !(optionHeight > 0)) continue;
    const diff = Math.abs(optionWidth / optionHeight - target);
    if (diff < closestDiff) {
      closestDiff = diff;
      closest = option;
    }
  }
  return closest;
}
