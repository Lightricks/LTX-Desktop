import { fromGeneration } from "../screens/Feature/definitions/audioToVideo.ts";

export type ExploreAudioToVideoSeedPlan = "wait" | "skip" | "ingest";

export const EXPLORE_AUDIO_TO_VIDEO_SEED_CACHE = {
  staleTime: Number.POSITIVE_INFINITY,
  gcTime: Number.POSITIVE_INFINITY,
} as const;

export function planExploreAudioToVideoSeed(input: {
  hasStoredValues: boolean;
  generationsReady: boolean;
  generationsFailed: boolean;
  lastGenerationSpec: unknown;
}): ExploreAudioToVideoSeedPlan {
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
