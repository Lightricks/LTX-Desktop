import { canAttachFeaturePreviewSrc } from "../../../../components/home/homeFeaturePreviewMedia.ts";

export type ExampleEmptyPlayback = "video" | "poster";

/**
 * Feature empty-state examples ship as `.webm`. iPhone Safari cannot decode
 * that, so attaching `src` and showing a play control leaves a dead overlay.
 */
export function exampleEmptyPlayback(
  videoUrl: string,
  canPlayType: (mime: string) => string,
): ExampleEmptyPlayback {
  return canAttachFeaturePreviewSrc(videoUrl, canPlayType) ? "video" : "poster";
}
