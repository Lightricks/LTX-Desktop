import {
  EXPLORE_AUDIO_TO_VIDEO_SEED_AUDIO_FILENAME,
  EXPLORE_AUDIO_TO_VIDEO_SEED_IMAGE_FILENAME,
  EXPLORE_IMAGE_TO_VIDEO_SEED_FILENAME,
  EXPLORE_LORA_RECIPE_SEED_FILES,
  RETAKE_SEED_FILENAME,
  EXTEND_SEED_FILENAME,
  icLoraSeedFilename,
  LAYOUT_TO_RENDER_IMAGE_SEED_FILENAME,
  RESTORE_IMAGE_SEED_FILENAME,
} from "../../../shared/explore-seed-filenames";
import audioToVideoSeedAudioUrl from "../../../resources/explore-assets/audio-to-video-input-v2.mp3";
import audioToVideoSeedImageUrl from "../../../resources/explore-assets/audio-to-video-input.jpg";
import cinemagraphStartFrameUrl from "../../../resources/explore-assets/cinemagraph-start-frame.jpg";
import dollyInStartFrameUrl from "../../../resources/explore-assets/dolly-in-start-frame.jpg";
import dollyOutStartFrameUrl from "../../../resources/explore-assets/dolly-out-start-frame.jpg";
import imageToVideoStartFrameUrl from "../../../resources/explore-assets/image-to-video-input.jpg";
import jibDownStartFrameUrl from "../../../resources/explore-assets/jib-down-start-frame.jpg";
import jibUpStartFrameUrl from "../../../resources/explore-assets/jib-up-start-frame.jpg";
import retakeSeedVideoUrl from "../../../resources/explore-assets/retake-input.mp4";
import dayToNightSeedVideoUrl from "../../../resources/explore-assets/day-to-night-reference.mp4";
import alphaGenSeedVideoUrl from "../../../resources/explore-assets/alpha-gen-reference.mp4";
import deblurSeedVideoUrl from "../../../resources/explore-assets/deblur-reference.mp4";
import colorizationSeedVideoUrl from "../../../resources/explore-assets/colorization-reference.mp4";
import cleanPlateSeedVideoUrl from "../../../resources/explore-assets/clean-plate-reference.mp4";
import decompressionSeedVideoUrl from "../../../resources/explore-assets/decompression-reference.mp4";
import waterSimulationSeedVideoUrl from "../../../resources/explore-assets/water-simulation-reference.mp4";
import layoutToRenderSeedImageUrl from "../../../resources/explore-assets/layout-to-render-first-frame.jpg";
import layoutToRenderSeedVideoUrl from "../../../resources/explore-assets/layout-to-render-reference.mp4";
import restoreSeedImageUrl from "../../../resources/explore-assets/restore-first-frame.jpg";
import restoreSeedVideoUrl from "../../../resources/explore-assets/restore-reference.mp4";
import extendSeedVideoUrl from "../../../resources/explore-assets/extend-input-v3.mp4";
import transitionEndFrameUrl from "../../../resources/explore-assets/transition-end-frame.jpg";
import transitionStartFrameUrl from "../../../resources/explore-assets/transition-start-frame.jpg";
import vbvrStartFrameUrl from "../../../resources/explore-assets/vbvr-start-frame.jpg";
import audioToVideoExamplePosterUrl from "./audio-to-video/example-poster.webp";
import audioToVideoExampleVideoUrl from "./audio-to-video/example.webm";
import extendExamplePosterUrl from "./extend/example-poster.webp";
import extendExampleVideoUrl from "./extend/example.webm";
import imageToVideoExamplePosterUrl from "./image-to-video/example-poster.webp";
import imageToVideoExampleVideoUrl from "./image-to-video/example.webm";
import retakeExamplePosterUrl from "./retake/example-poster.webp";
import retakeExampleVideoUrl from "./retake/example.webm";
import textToVideoExamplePosterUrl from "./text-to-video/example-poster.webp";
import textToVideoExampleVideoUrl from "./text-to-video/example.webm";

/**
 * Ingestible packaged Explore seeds. Empty-state examples live in
 * `PACKAGED_EXPLORE_EXAMPLES` and are display URLs only.
 */
export const PACKAGED_EXPLORE_SEEDS = {
  "image-to-video-start-frame": {
    filename: EXPLORE_IMAGE_TO_VIDEO_SEED_FILENAME,
    url: imageToVideoStartFrameUrl,
    mimeType: "image/jpeg",
  },
  "cinemagraph-start-frame": {
    filename: EXPLORE_LORA_RECIPE_SEED_FILES["cinemagraph-start-frame"],
    url: cinemagraphStartFrameUrl,
    mimeType: "image/jpeg",
  },
  "jib-up-start-frame": {
    filename: EXPLORE_LORA_RECIPE_SEED_FILES["jib-up-start-frame"],
    url: jibUpStartFrameUrl,
    mimeType: "image/jpeg",
  },
  "jib-down-start-frame": {
    filename: EXPLORE_LORA_RECIPE_SEED_FILES["jib-down-start-frame"],
    url: jibDownStartFrameUrl,
    mimeType: "image/jpeg",
  },
  "dolly-in-start-frame": {
    filename: EXPLORE_LORA_RECIPE_SEED_FILES["dolly-in-start-frame"],
    url: dollyInStartFrameUrl,
    mimeType: "image/jpeg",
  },
  "dolly-out-start-frame": {
    filename: EXPLORE_LORA_RECIPE_SEED_FILES["dolly-out-start-frame"],
    url: dollyOutStartFrameUrl,
    mimeType: "image/jpeg",
  },
  "transition-start-frame": {
    filename: EXPLORE_LORA_RECIPE_SEED_FILES["transition-start-frame"],
    url: transitionStartFrameUrl,
    mimeType: "image/jpeg",
  },
  "transition-end-frame": {
    filename: EXPLORE_LORA_RECIPE_SEED_FILES["transition-end-frame"],
    url: transitionEndFrameUrl,
    mimeType: "image/jpeg",
  },
  "vbvr-start-frame": {
    filename: EXPLORE_LORA_RECIPE_SEED_FILES["vbvr-start-frame"],
    url: vbvrStartFrameUrl,
    mimeType: "image/jpeg",
  },
  "audio-to-video-audio": {
    filename: EXPLORE_AUDIO_TO_VIDEO_SEED_AUDIO_FILENAME,
    url: audioToVideoSeedAudioUrl,
    mimeType: "audio/mpeg",
  },
  "audio-to-video-start-frame": {
    filename: EXPLORE_AUDIO_TO_VIDEO_SEED_IMAGE_FILENAME,
    url: audioToVideoSeedImageUrl,
    mimeType: "image/jpeg",
  },
  "retake-video": {
    filename: RETAKE_SEED_FILENAME,
    url: retakeSeedVideoUrl,
    mimeType: "video/mp4",
  },
  "extend-video": {
    filename: EXTEND_SEED_FILENAME,
    url: extendSeedVideoUrl,
    mimeType: "video/mp4",
  },
  "day-to-night-video": {
    filename: icLoraSeedFilename("day-to-night"),
    url: dayToNightSeedVideoUrl,
    mimeType: "video/mp4",
  },
  "alpha-gen-video": {
    filename: icLoraSeedFilename("alpha-gen"),
    url: alphaGenSeedVideoUrl,
    mimeType: "video/mp4",
  },
  "deblur-video": {
    filename: icLoraSeedFilename("deblur"),
    url: deblurSeedVideoUrl,
    mimeType: "video/mp4",
  },
  "colorization-video": {
    filename: icLoraSeedFilename("colorization"),
    url: colorizationSeedVideoUrl,
    mimeType: "video/mp4",
  },
  "clean-plate-video": {
    filename: icLoraSeedFilename("clean-plate"),
    url: cleanPlateSeedVideoUrl,
    mimeType: "video/mp4",
  },
  "decompression-video": {
    filename: icLoraSeedFilename("decompression"),
    url: decompressionSeedVideoUrl,
    mimeType: "video/mp4",
  },
  "water-simulation-video": {
    filename: icLoraSeedFilename("water-simulation"),
    url: waterSimulationSeedVideoUrl,
    mimeType: "video/mp4",
  },
  "layout-to-render-video": {
    filename: icLoraSeedFilename("layout-to-render"),
    url: layoutToRenderSeedVideoUrl,
    mimeType: "video/mp4",
  },
  "layout-to-render-image": {
    filename: LAYOUT_TO_RENDER_IMAGE_SEED_FILENAME,
    url: layoutToRenderSeedImageUrl,
    mimeType: "image/jpeg",
  },
  "restore-video": {
    filename: icLoraSeedFilename("restore"),
    url: restoreSeedVideoUrl,
    mimeType: "video/mp4",
  },
  "restore-image": {
    filename: RESTORE_IMAGE_SEED_FILENAME,
    url: restoreSeedImageUrl,
    mimeType: "image/jpeg",
  },
} as const;

export type PackagedExploreSeedId = keyof typeof PACKAGED_EXPLORE_SEEDS;

export const PACKAGED_EXPLORE_EXAMPLES = {
  "image-to-video-example-poster": {
    filename: "example-poster.webp",
    url: imageToVideoExamplePosterUrl,
    mimeType: "image/webp",
  },
  "image-to-video-example-video": {
    filename: "example.webm",
    url: imageToVideoExampleVideoUrl,
    mimeType: "video/webm",
  },
  "text-to-video-example-poster": {
    filename: "example-poster.webp",
    url: textToVideoExamplePosterUrl,
    mimeType: "image/webp",
  },
  "text-to-video-example-video": {
    filename: "example.webm",
    url: textToVideoExampleVideoUrl,
    mimeType: "video/webm",
  },
  "audio-to-video-example-poster": {
    filename: "example-poster.webp",
    url: audioToVideoExamplePosterUrl,
    mimeType: "image/webp",
  },
  "audio-to-video-example-video": {
    filename: "example.webm",
    url: audioToVideoExampleVideoUrl,
    mimeType: "video/webm",
  },
  "retake-example-poster": {
    filename: "example-poster.webp",
    url: retakeExamplePosterUrl,
    mimeType: "image/webp",
  },
  "retake-example-video": {
    filename: "example.webm",
    url: retakeExampleVideoUrl,
    mimeType: "video/webm",
  },
  "extend-example-poster": {
    filename: "example-poster.webp",
    url: extendExamplePosterUrl,
    mimeType: "image/webp",
  },
  "extend-example-video": {
    filename: "example.webm",
    url: extendExampleVideoUrl,
    mimeType: "video/webm",
  },
} as const;

export type PackagedExploreExampleId = keyof typeof PACKAGED_EXPLORE_EXAMPLES;

export function packagedExploreExampleUrl(id: PackagedExploreExampleId): string {
  return PACKAGED_EXPLORE_EXAMPLES[id].url;
}

export async function loadPackagedExploreSeed(
  id: PackagedExploreSeedId,
): Promise<File> {
  const spec = PACKAGED_EXPLORE_SEEDS[id];
  const response = await fetch(spec.url);
  if (!response.ok) {
    throw new Error(`Packaged Explore seed "${id}" is missing`);
  }
  const blob = await response.blob();
  const mimePrefix = spec.mimeType.slice(0, spec.mimeType.indexOf("/") + 1);
  const type = blob.type.startsWith(mimePrefix) ? blob.type : spec.mimeType;
  return new File([blob], spec.filename, { type });
}
