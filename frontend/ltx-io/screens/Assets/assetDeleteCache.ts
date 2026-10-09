import {
  type InfiniteData,
  type QueryClient,
  type QueryKey,
} from "@tanstack/react-query";

import type { ExploreAssetListResponse } from "../../../lib/explore-contract.ts";

import { assetQueryKeys } from "../../hooks/assetQueryKeys.ts";

export type AssetListInfiniteData = InfiniteData<
  ExploreAssetListResponse,
  string | null
>;
export type AssetCacheSnapshot = readonly [
  QueryKey,
  AssetListInfiniteData | undefined,
][];

export const assetListQueryFilter = {
  queryKey: [...assetQueryKeys.all, "list"] as const,
};

export function removeAssetFromInfinitePages(
  data: AssetListInfiniteData,
  assetId: string,
): AssetListInfiniteData {
  return {
    ...data,
    pages: data.pages.map((page) => ({
      ...page,
      items: page.items.filter((asset) => asset.id !== assetId),
    })),
  };
}

export async function optimisticallyRemoveAssetFromListCache(
  queryClient: QueryClient,
  assetId: string,
): Promise<AssetCacheSnapshot> {
  await queryClient.cancelQueries(assetListQueryFilter);
  const snapshots = queryClient.getQueriesData<AssetListInfiniteData>(
    assetListQueryFilter,
  );
  queryClient.setQueriesData<AssetListInfiniteData>(
    assetListQueryFilter,
    (current) =>
      current === undefined
        ? current
        : removeAssetFromInfinitePages(current, assetId),
  );
  return snapshots;
}

export function restoreAssetCache(
  queryClient: QueryClient,
  snapshots: AssetCacheSnapshot,
): void {
  for (const [queryKey, snapshot] of snapshots) {
    queryClient.setQueryData(queryKey, snapshot);
  }
}
