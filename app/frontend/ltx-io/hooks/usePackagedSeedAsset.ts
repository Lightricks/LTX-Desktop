import {
  queryOptions,
  useQueries,
  useQuery,
  useQueryClient,
  type QueryClient,
} from "@tanstack/react-query";

import { toDurableAssetRef, type AssetRef } from "../screens/Feature/types";
import { useExploreRuntime, type ExploreRuntime } from "../runtime/ExploreRuntime";
import type { PackagedExploreSeedId } from "../assets/packaged-explore-assets";
import { assetQueryKeys, isImageAsset, isVideoAsset } from "./assetQueryKeys";
import { PACKAGED_SEED_CACHE } from "./packagedSeed.ts";

type PackagedSeedOptions = {
  enabled?: boolean;
  revision?: number;
  media: "video" | "image";
};

function packagedSeedQuery(
  id: PackagedExploreSeedId,
  options: PackagedSeedOptions,
  loadPackagedAsset: ExploreRuntime["loadPackagedAsset"],
  queryClient: QueryClient,
) {
  const { media } = options;
  return queryOptions({
    // `media` is in the key: one seed kind can be read as a video and as an image.
    queryKey: ["packaged-seed", id, media, options.revision ?? 0] as const,
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
    enabled: options.enabled ?? true,
    ...PACKAGED_SEED_CACHE,
  });
}

export function usePackagedSeedAsset(
  id: PackagedExploreSeedId,
  options: PackagedSeedOptions = { media: "video" },
) {
  const { loadPackagedAsset } = useExploreRuntime();
  const queryClient = useQueryClient();
  return useQuery(packagedSeedQuery(id, options, loadPackagedAsset, queryClient));
}

function combineSeedAssets(results: { data?: AssetRef; error: Error | null }[]) {
  const data = results.every((result) => result.data != null)
    ? results.map((result) => result.data as AssetRef)
    : undefined;
  return {
    data,
    isSuccess: data != null,
    error: results.find((result) => result.error != null)?.error ?? null,
  };
}

/** Loads several packaged seeds of one media kind. `data` keeps the order of `ids`. */
export function usePackagedSeedAssets(
  ids: readonly PackagedExploreSeedId[],
  options: PackagedSeedOptions,
) {
  const { loadPackagedAsset } = useExploreRuntime();
  const queryClient = useQueryClient();
  return useQueries({
    queries: ids.map((id) => packagedSeedQuery(id, options, loadPackagedAsset, queryClient)),
    combine: combineSeedAssets,
  });
}
