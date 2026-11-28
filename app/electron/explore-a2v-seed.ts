import path from "node:path";

import {
  EXPLORE_AUDIO_TO_VIDEO_SEED_AUDIO_FILENAME,
  EXPLORE_AUDIO_TO_VIDEO_SEED_DIR,
  EXPLORE_AUDIO_TO_VIDEO_SEED_IMAGE_FILENAME,
} from "../shared/explore-seed-filenames.ts";

export {
  EXPLORE_AUDIO_TO_VIDEO_SEED_AUDIO_FILENAME,
  EXPLORE_AUDIO_TO_VIDEO_SEED_DIR,
  EXPLORE_AUDIO_TO_VIDEO_SEED_IMAGE_FILENAME,
};

export type ExploreAudioToVideoSeedPaths = {
  audio: string;
  startFrame: string;
};

export function resolveExploreAudioToVideoSeedPaths(input: {
  isPackaged: boolean;
  projectRoot: string;
  resourcesPath: string;
}): ExploreAudioToVideoSeedPaths {
  const seedDir = input.isPackaged
    ? path.join(input.resourcesPath, EXPLORE_AUDIO_TO_VIDEO_SEED_DIR)
    : path.join(input.projectRoot, "resources", EXPLORE_AUDIO_TO_VIDEO_SEED_DIR);
  return {
    audio: path.join(seedDir, EXPLORE_AUDIO_TO_VIDEO_SEED_AUDIO_FILENAME),
    startFrame: path.join(seedDir, EXPLORE_AUDIO_TO_VIDEO_SEED_IMAGE_FILENAME),
  };
}
