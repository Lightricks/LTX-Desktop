import { useQuery, useQueryClient } from "@tanstack/react-query";

import { toDurableAssetRef } from "../screens/Feature/types";
import { useExploreRuntime } from "../runtime/ExploreRuntime";
import type { PackagedExploreSeedId } from "../assets/packaged-explore-assets";
import { assetQueryKeys, isImageAsset, isVideoAsset } from "./assetQueryKeys";
import { PACKAGED_SEED_CACHE } from "./packagedSeed.ts";

export function usePackagedSeedAsset(
  id: PackagedExploreSeedId,
  options?: {
    enabled?: boolean;
    revision?: number;
    media: "video" | "image";
  },
) {
  const { loadPackagedAsset } = useExploreRuntime();
  const queryClient = useQueryClient();
  const media = options?.media ?? "video";

  return useQuery({
    // `media` is in the key: one seed kind can be read as a video and as an image.
    queryKey: ["packaged-seed", id, media, options?.revision ?? 0] as const,
    queryFn: async () => {
      const asset = await loadPackagedAsset(id);
      if (media === "video" && !isVideoAsset(asset)) {
        throw new Error(`${id} seed must be a video`);
      }
      if (media === "image" && !isImageAsset(asset)) {
        throw new Error(`${id} seed must be an image`);
      }
      queryClient.setQueryData(assetQueryKeys.detail(asset.id), asset);
      return toDurableAssetRef(asset);
    },
    enabled: options?.enabled ?? true,
    ...PACKAGED_SEED_CACHE,
  });
}
