import { useQueries } from "@tanstack/react-query";

import type { ExploreListedAsset } from "@/lib/explore-contract";

import { assetQueryKeys } from "../../../../hooks/assetQueryKeys";
import { fetchAsset } from "../../../../hooks/generationQueryKeys";
import { useExploreRuntime } from "../../../../runtime/ExploreRuntime";

import { orderLightboxAssets } from "./keyframesModel";

/**
 * The assets of the stills for the lightbox, with the place of each still in the list.
 * The queries share the cache with `useAsset`, so a thumbnail that already loaded costs no new
 * request.
 */
export function useKeyframeLightboxAssets(assetIds: readonly string[]): {
  items: ExploreListedAsset[];
  positions: (number | null)[];
} {
  const { api } = useExploreRuntime();
  const uniqueIds = [...new Set(assetIds)];
  const results = useQueries({
    queries: uniqueIds.map((assetId) => ({
      queryKey: assetQueryKeys.detail(assetId),
      queryFn: () => fetchAsset(api, assetId),
    })),
  });
  const loaded = new Map<string, ExploreListedAsset>();
  uniqueIds.forEach((assetId, index) => {
    const asset = results[index]?.data;
    if (asset) loaded.set(assetId, { ...asset, in_use: true });
  });
  return orderLightboxAssets(assetIds, loaded);
}
