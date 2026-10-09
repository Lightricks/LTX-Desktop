import { useEffect, useMemo } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import {
  isHomeFeatureEnabled,
  isHomeFeatureId,
  type HomeFeatureId,
} from "../../lib/home-features.ts";
import { MAX_SIDEBAR_RECENT_ITEMS } from "../../lib/home-sidebar-recents.ts";
import { generationQueryKeys } from "../../ltx-io/hooks/generationQueryKeys.ts";
import { unwrapApiResult } from "../../ltx-io/lib/unwrapApiResult.ts";
import { useExploreRuntime } from "../../ltx-io/runtime/ExploreRuntime.tsx";

/** Only refetch on return if the list is older than this; local actions already invalidate it. */
export const RECENT_SIDEBAR_TOOLS_STALE_MS = 30_000;

const RECENT_FEATURES_QUERY_KEY = generationQueryKeys.recentFeatures(
  MAX_SIDEBAR_RECENT_ITEMS,
);

/** Distinct features from the newest generations, shared by desktop and remote. */
export function useRecentSidebarTools(): readonly HomeFeatureId[] {
  const { api } = useExploreRuntime();
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: RECENT_FEATURES_QUERY_KEY,
    queryFn: async () =>
      unwrapApiResult(
        await api.listRecentFeatures({ limit: MAX_SIDEBAR_RECENT_ITEMS }),
      ),
    staleTime: RECENT_SIDEBAR_TOOLS_STALE_MS,
    // React Query only reacts to tab visibility (the remote app); the window
    // focus listener below covers switching back to the Electron window.
    refetchOnWindowFocus: true,
  });

  useEffect(() => {
    const refetchIfStale = () => {
      void queryClient.refetchQueries({
        queryKey: RECENT_FEATURES_QUERY_KEY,
        type: "active",
        stale: true,
      });
    };
    window.addEventListener("focus", refetchIfStale);
    return () => window.removeEventListener("focus", refetchIfStale);
  }, [queryClient]);

  const features = query.data;
  return useMemo(
    () =>
      (features ?? []).filter(
        (id): id is HomeFeatureId => isHomeFeatureId(id) && isHomeFeatureEnabled(id),
      ),
    [features],
  );
}
