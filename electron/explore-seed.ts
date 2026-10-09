import path from "node:path";

import {
  EXPLORE_IMAGE_TO_VIDEO_SEED_DIR,
  EXPLORE_IMAGE_TO_VIDEO_SEED_FILENAME,
  EXPLORE_LORA_RECIPE_SEED_DIR,
  EXPLORE_LORA_RECIPE_SEED_FILES,
  EXTEND_SEED_DIR,
  EXTEND_SEED_FILENAME,
  IC_LORA_SEED_DIR,
  icLoraSeedFilename,
  LAYOUT_TO_RENDER_IMAGE_SEED_FILENAME,
  PACKAGED_SEED_KINDS,
  RESTORE_IMAGE_SEED_FILENAME,
  RETAKE_SEED_DIR,
  RETAKE_SEED_FILENAME,
  type PackagedSeedKind,
} from "../shared/explore-seed-filenames.ts";

export { PACKAGED_SEED_KINDS, type PackagedSeedKind };

function icLoraSeedSpec(recipeId: string, title: string) {
  return {
    dir: IC_LORA_SEED_DIR,
    filename: icLoraSeedFilename(recipeId),
    missing: `${title} reference video is missing`,
  };
}

const PACKAGED_SEEDS: Record<
  PackagedSeedKind,
  { dir: string; filename: string; missing: string }
> = {
  "image-to-video-start-frame": {
    dir: EXPLORE_IMAGE_TO_VIDEO_SEED_DIR,
    filename: EXPLORE_IMAGE_TO_VIDEO_SEED_FILENAME,
    missing: "Explore image-to-video seed is missing",
  },
  "dolly-in-start-frame": {
    dir: EXPLORE_LORA_RECIPE_SEED_DIR,
    filename: EXPLORE_LORA_RECIPE_SEED_FILES["dolly-in-start-frame"],
    missing: "Dolly In start-frame seed is missing",
  },
  "retake-video": {
    dir: RETAKE_SEED_DIR,
    filename: RETAKE_SEED_FILENAME,
    missing: "Retake seed is missing",
  },
  "extend-video": {
    dir: EXTEND_SEED_DIR,
    filename: EXTEND_SEED_FILENAME,
    missing: "Extend seed is missing",
  },
  "day-to-night-video": icLoraSeedSpec("day-to-night", "Day to Night"),
  "alpha-gen-video": icLoraSeedSpec("alpha-gen", "AlphaGen"),
  "deblur-video": icLoraSeedSpec("deblur", "Deblur"),
  "colorization-video": icLoraSeedSpec("colorization", "Colorization"),
  "clean-plate-video": icLoraSeedSpec("clean-plate", "Clean Plate"),
  "decompression-video": icLoraSeedSpec("decompression", "Decompression"),
  "water-simulation-video": icLoraSeedSpec("water-simulation", "Water Simulation"),
  "layout-to-render-video": icLoraSeedSpec("layout-to-render", "Layout to Render"),
  "layout-to-render-image": {
    dir: IC_LORA_SEED_DIR,
    filename: LAYOUT_TO_RENDER_IMAGE_SEED_FILENAME,
    missing: "Layout to Render look image is missing",
  },
  "restore-video": icLoraSeedSpec("restore", "Restore"),
  "restore-image": {
    dir: IC_LORA_SEED_DIR,
    filename: RESTORE_IMAGE_SEED_FILENAME,
    missing: "Restore look image is missing",
  },
};

export function packagedSeedSpec(kind: PackagedSeedKind) {
  return PACKAGED_SEEDS[kind];
}

export function packagedSeedMissingMessage(kind: PackagedSeedKind): string {
  return PACKAGED_SEEDS[kind].missing;
}

export function resolvePackagedSeedPath(input: {
  kind: PackagedSeedKind;
  isPackaged: boolean;
  projectRoot: string;
  resourcesPath: string;
}): string {
  const spec = PACKAGED_SEEDS[input.kind];
  const segments = [spec.dir, spec.filename] as const;
  if (input.isPackaged) {
    return path.join(input.resourcesPath, ...segments);
  }
  return path.join(input.projectRoot, "resources", ...segments);
}
