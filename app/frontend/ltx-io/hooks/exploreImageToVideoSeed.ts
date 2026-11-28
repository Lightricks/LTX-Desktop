import { fromGeneration } from "../screens/Feature/definitions/imageToVideo.ts";

export type ExploreImageToVideoSeedPlan = "wait" | "skip" | "ingest";

export const EXPLORE_IMAGE_TO_VIDEO_SEED_CACHE = {
  staleTime: Number.POSITIVE_INFINITY,
  gcTime: Number.POSITIVE_INFINITY,
} as const;

export function planExploreImageToVideoSeed(input: {
  hasStoredValues: boolean;
  generationsReady: boolean;
  generationsFailed: boolean;
  lastGenerationSpec: unknown;
}): ExploreImageToVideoSeedPlan {
  if (input.hasStoredValues || input.generationsFailed) {
    return "skip";
  }
  if (!input.generationsReady) {
    return "wait";
  }
  if (fromGeneration(input.lastGenerationSpec) != null) {
    return "skip";
  }
  return "ingest";
}
