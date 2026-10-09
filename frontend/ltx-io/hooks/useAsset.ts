import { useQuery } from "@tanstack/react-query";

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
