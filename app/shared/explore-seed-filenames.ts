// Pure filename constants shared by Electron (path ingest) and the Vite
// catalog (bundled bytes). No node: or Vite imports allowed here.
export const EXPLORE_IMAGE_TO_VIDEO_SEED_DIR = "explore-assets";
export const EXPLORE_IMAGE_TO_VIDEO_SEED_FILENAME = "image-to-video-input.jpg";

export const EXPLORE_AUDIO_TO_VIDEO_SEED_DIR = "explore-assets";
export const EXPLORE_AUDIO_TO_VIDEO_SEED_AUDIO_FILENAME =
  "audio-to-video-input-v2.mp3";
export const EXPLORE_AUDIO_TO_VIDEO_SEED_IMAGE_FILENAME =
  "audio-to-video-input.jpg";

export const EXPLORE_LORA_RECIPE_SEED_DIR = "explore-assets";

export const EXPLORE_LORA_RECIPE_SEED_IDS = [
  "cinemagraph-start-frame",
  "jib-up-start-frame",
  "jib-down-start-frame",
  "dolly-in-start-frame",
  "dolly-out-start-frame",
  "transition-start-frame",
  "transition-end-frame",
  "vbvr-start-frame",
] as const;

export type ExploreLoraRecipeSeedId = (typeof EXPLORE_LORA_RECIPE_SEED_IDS)[number];

export const EXPLORE_LORA_RECIPE_SEED_FILES: Record<
  ExploreLoraRecipeSeedId,
  string
> = {
  "cinemagraph-start-frame": "cinemagraph-start-frame.jpg",
  "jib-up-start-frame": "jib-up-start-frame.jpg",
  "jib-down-start-frame": "jib-down-start-frame.jpg",
  "dolly-in-start-frame": "dolly-in-start-frame.jpg",
  "dolly-out-start-frame": "dolly-out-start-frame.jpg",
  "transition-start-frame": "transition-start-frame.jpg",
  "transition-end-frame": "transition-end-frame.jpg",
  "vbvr-start-frame": "vbvr-start-frame.jpg",
};

export const RETAKE_SEED_DIR = EXPLORE_IMAGE_TO_VIDEO_SEED_DIR;
export const RETAKE_SEED_FILENAME = "retake-input.mp4";
/** Previous packaged names still sitting in app-data from earlier builds. */
export const RETAKE_SEED_PREVIOUS_FILENAMES = ["retake-input-v2.mp4"] as const;
/**
 * Bump when the packaged retake clip changes so a hydrated example re-ingests
 * instead of keeping the previous copy from Application Support.
 */
export const RETAKE_SEED_REVISION = 5;

export const EXTEND_SEED_DIR = EXPLORE_IMAGE_TO_VIDEO_SEED_DIR;
export const EXTEND_SEED_FILENAME = "extend-input-v3.mp4";
/** Bump when the packaged extend clip changes so a hydrated example re-ingests. */
export const EXTEND_SEED_REVISION = 2;
export const EXTEND_SEED_PREVIOUS_FILENAMES = [] as const;

export const IC_LORA_SEED_DIR = EXPLORE_IMAGE_TO_VIDEO_SEED_DIR;
/** Packaged IC-LoRA reference clip. One file per recipe: `<recipe id>-reference.mp4`. */
export function icLoraSeedFilename(recipeId: string): string {
  return `${recipeId}-reference.mp4`;
}

/** The finished first-frame image that sets the look. It is not a clip frame. */
export const LAYOUT_TO_RENDER_IMAGE_SEED_FILENAME = "layout-to-render-first-frame.jpg";
export const LAYOUT_TO_RENDER_IMAGE_SEED_PREVIOUS_FILENAMES = [] as const;
/** Bump when the packaged look image changes, so a stored copy is replaced. */
export const LAYOUT_TO_RENDER_IMAGE_SEED_REVISION = 1;
/** The restored first frame of the packaged Restore clip. It sets the look of the result. */
export const RESTORE_IMAGE_SEED_FILENAME = "restore-first-frame.jpg";
export const RESTORE_IMAGE_SEED_PREVIOUS_FILENAMES = [] as const;
/** Bump when the packaged look image changes, so a stored copy is replaced. */
export const RESTORE_IMAGE_SEED_REVISION = 1;
/** The pre-made reference sheet that fills the Ingredients form at first open. */
export const INGREDIENTS_SHEET_SEED_FILENAME = "ingredients-sheet.jpg";
/** The still that fills the Motion Track form at first open. Same file as ltx.io. */
export const MOTION_TRACK_IMAGE_SEED = {
  kind: "motion-track-image",
  filename: "motion-track-input.jpg",
} as const;
/** The start frame that fills the Video Guide form at first open. Same file as ltx.io. */
export const VIDEO_GUIDE_IMAGE_SEED = {
  kind: "video-guide-image",
  filename: "video-guide-start-frame.jpg",
  previousFilenames: [],
  /** Bump when the packaged start frame changes, so a stored copy is replaced. */
  revision: 1,
} as const;
/**
 * The four season stills that fill the Keyframes form at first open. The summer still
 * sits at both the first and the last slot, so it is packaged once.
 */
export const KEYFRAMES_SEED_KINDS = [
  "keyframes-summer-image",
  "keyframes-autumn-image",
  "keyframes-winter-image",
  "keyframes-spring-image",
] as const;
export type KeyframesSeedKind = (typeof KEYFRAMES_SEED_KINDS)[number];
export const KEYFRAMES_SEED_FILES: Record<KeyframesSeedKind, string> = {
  "keyframes-summer-image": "keyframes-summer.jpg",
  "keyframes-autumn-image": "keyframes-autumn.jpg",
  "keyframes-winter-image": "keyframes-winter.jpg",
  "keyframes-spring-image": "keyframes-spring.jpg",
};

/** Single-file packaged seeds resolved by `getPackagedSeedPath({ kind })`. */
export const PACKAGED_SEED_KINDS = [
  "image-to-video-start-frame",
  "dolly-in-start-frame",
  "retake-video",
  "extend-video",
  "day-to-night-video",
  "alpha-gen-video",
  "deblur-video",
  "colorization-video",
  "clean-plate-video",
  "decompression-video",
  "water-simulation-video",
  "layout-to-render-video",
  "layout-to-render-image",
  "restore-video",
  "restore-image",
  "ingredients-sheet",
  "refine-details-video",
  "motion-track-image",
  "video-guide-video",
  "video-guide-image",
  ...KEYFRAMES_SEED_KINDS,
] as const;
export type PackagedSeedKind = (typeof PACKAGED_SEED_KINDS)[number];
