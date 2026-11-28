import { useQuery, useQueryClient } from "@tanstack/react-query";

import { toDurableAssetRef } from "../screens/Feature/types";
import { useExploreRuntime } from "../runtime/ExploreRuntime";
import type { PackagedExploreSeedId } from "../assets/packaged-explore-assets";
import { assetQueryKeys, isImageAsset } from "./assetQueryKeys";
import { EXPLORE_IMAGE_TO_VIDEO_SEED_CACHE } from "./exploreImageToVideoSeed.ts";

export function exploreLoraRecipeSeedQueryKey(
  seedId: PackagedExploreSeedId | undefined,
) {
  return ["explore-lora-recipe-seed", seedId] as const;
}

export function useExploreLoraRecipeSeed(options: {
  seedId: PackagedExploreSeedId | undefined;
  enabled?: boolean;
}) {
  const { loadPackagedAsset } = useExploreRuntime();
  const queryClient = useQueryClient();
  const seedId = options.seedId;

  return useQuery({
    queryKey: exploreLoraRecipeSeedQueryKey(seedId),
    queryFn: async () => {
      if (seedId == null) {
        throw new Error("LoRA recipe start-frame seed id is missing");
      }
      const asset = await loadPackagedAsset(seedId);
      if (!isImageAsset(asset)) {
        throw new Error("LoRA recipe start-frame seed must be an image");
      }
      queryClient.setQueryData(assetQueryKeys.detail(asset.id), asset);
      return toDurableAssetRef(asset);
    },
    enabled: seedId != null && (options.enabled ?? true),
    ...EXPLORE_IMAGE_TO_VIDEO_SEED_CACHE,
  });
}
