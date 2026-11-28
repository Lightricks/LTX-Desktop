import { useInfiniteQuery } from "@tanstack/react-query";
import { useCallback } from "react";
import { useSearchParams } from "react-router";

import { assetQueryKeys, canonicalAssetSearchQuery } from "../../hooks/assetQueryKeys";
import { useInvalidateAssetsOnGenerationSuccess } from "../../hooks/useInvalidateAssetsOnGenerationSuccess";
import { unwrapApiResult } from "../../lib/unwrapApiResult";
import { useExploreRuntime } from "../../runtime/ExploreRuntime";

import {
  type AssetsListFilters,
  filtersFromSearchParams,
  searchParamsFromFilters,
} from "./assetsSearchParams";

export function useAssetsList() {
  const [searchParams, setSearchParams] = useSearchParams();
  const { api } = useExploreRuntime();
  useInvalidateAssetsOnGenerationSuccess();
  const filters = filtersFromSearchParams(searchParams);
  const media_kind = filters.mediaKind ?? undefined;
  const sort = filters.sort;
  const q = canonicalAssetSearchQuery(filters.q);
  const listFilters = {
    media_kind,
    sort,
    ...(q === undefined ? {} : { q }),
  };

  const query = useInfiniteQuery({
    queryKey: assetQueryKeys.list(listFilters),
    initialPageParam: null as string | null,
    queryFn: async ({ pageParam }) =>
      unwrapApiResult(
        await api.listAssets({
          ...listFilters,
          cursor: pageParam,
        }),
      ),
    getNextPageParam: (last) => last.next_cursor ?? undefined,
  });

  const setFilters = useCallback(
    (next: AssetsListFilters) => {
      setSearchParams(searchParamsFromFilters(next), { replace: true });
    },
    [setSearchParams],
  );

  return { ...query, filters, setFilters };
}
