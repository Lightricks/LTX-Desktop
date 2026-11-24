import { useQuery, useQueryClient } from "@tanstack/react-query";

import { toDurableAssetRef } from "../screens/Feature/types";
import type { AudioToVideoSeed } from "../screens/Feature/definitions/audioToVideo";
import { useExploreRuntime } from "../runtime/ExploreRuntime";
import { assetQueryKeys, isAudioAsset, isImageAsset } from "./assetQueryKeys";
import { EXPLORE_AUDIO_TO_VIDEO_SEED_CACHE } from "./exploreAudioToVideoSeed.ts";

export const exploreAudioToVideoSeedQueryKey = [
  "explore-audio-to-video-seed",
] as const;

export function useExploreAudioToVideoSeed(options?: { enabled?: boolean }) {
  const { loadPackagedAsset } = useExploreRuntime();
  const queryClient = useQueryClient();

  return useQuery<AudioToVideoSeed>({
    queryKey: exploreAudioToVideoSeedQueryKey,
    queryFn: async () => {
      const [audio, startFrame] = await Promise.all([
        loadPackagedAsset("audio-to-video-audio"),
        loadPackagedAsset("audio-to-video-start-frame"),
      ]);
      if (!isAudioAsset(audio)) {
        throw new Error("Explore audio-to-video seed must be an audio file");
      }
      if (!isImageAsset(startFrame)) {
        throw new Error("Explore audio-to-video seed image must be an image");
      }
      queryClient.setQueryData(assetQueryKeys.detail(audio.id), audio);
      queryClient.setQueryData(
        assetQueryKeys.detail(startFrame.id),
        startFrame,
      );
      return {
        audio: toDurableAssetRef(audio),
        startFrame: toDurableAssetRef(startFrame),
      };
    },
    enabled: options?.enabled ?? true,
    ...EXPLORE_AUDIO_TO_VIDEO_SEED_CACHE,
  });
}
