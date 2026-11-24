import { useQuery, useQueryClient } from "@tanstack/react-query";

import { toDurableAssetRef } from "../screens/Feature/types";
import { useExploreRuntime } from "../runtime/ExploreRuntime";
import { assetQueryKeys, isImageAsset } from "./assetQueryKeys";
import { EXPLORE_IMAGE_TO_VIDEO_SEED_CACHE } from "./exploreImageToVideoSeed.ts";

export const exploreImageToVideoSeedQueryKey = [
  "explore-image-to-video-seed",
] as const;

export function useExploreImageToVideoSeed(options?: { enabled?: boolean }) {
  const { loadPackagedAsset } = useExploreRuntime();
  const queryClient = useQueryClient();

  return useQuery({
    queryKey: exploreImageToVideoSeedQueryKey,
    queryFn: async () => {
      const asset = await loadPackagedAsset("image-to-video-start-frame");
      if (!isImageAsset(asset)) {
        throw new Error("Explore image-to-video seed must be an image");
      }
      queryClient.setQueryData(assetQueryKeys.detail(asset.id), asset);
      return toDurableAssetRef(asset);
    },
    enabled: options?.enabled ?? true,
    ...EXPLORE_IMAGE_TO_VIDEO_SEED_CACHE,
  });
}
