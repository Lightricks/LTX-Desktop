import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import { useExploreRuntime } from "../runtime/ExploreRuntime";
import { assetQueryKeys } from "./assetQueryKeys";
import { fetchAsset } from "./generationQueryKeys";

export function useAsset(assetId: string | null) {
  const { api } = useExploreRuntime();
  return useQuery({
    queryKey: assetQueryKeys.detail(assetId ?? ""),
    queryFn: () => fetchAsset(api, assetId as string),
    enabled: Boolean(assetId),
  });
}

/** The pixel size of an image asset. Null while it loads, when there is no asset, or for a non-image. */
export function useImageAssetSize(assetId: string | null): { width: number; height: number } | null {
  const metadata = useAsset(assetId).data?.metadata;
  return useMemo(
    () =>
      metadata?.mediaType === "image"
        ? { width: metadata.metadata.width, height: metadata.metadata.height }
        : null,
    [metadata],
  );
}
