import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import type { components } from "../../generated/backend-openapi";
import { unwrapApiResult } from "../lib/unwrapApiResult";
import type { ExploreApi } from "../runtime/ExploreRuntime";

type FeatureFlagsPatch = components["schemas"]["FeatureFlagsPatch"];

/** Takes the bound client so it works on Desktop (`ApiClient`) and Remote (`runtime.api`). */
export type FeatureFlagsApi = Pick<ExploreApi, "getFeatureFlags" | "updateFeatureFlags">;

const FEATURE_FLAGS_QUERY_KEY = ["feature-flags"] as const;

export function useFeatureFlags(api: FeatureFlagsApi) {
  return useQuery({
    queryKey: FEATURE_FLAGS_QUERY_KEY,
    queryFn: async () => unwrapApiResult(await api.getFeatureFlags()),
    // The shared client doesn't retry. Flags are fetched once at app start and gate
    // features, so ride out a backend that is still coming up.
    retry: 2,
  });
}

export function useUpdateFeatureFlags(api: FeatureFlagsApi) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (patch: FeatureFlagsPatch) =>
      unwrapApiResult(await api.updateFeatureFlags(patch)),
    onSuccess: (flags) => {
      queryClient.setQueryData(FEATURE_FLAGS_QUERY_KEY, flags);
    },
  });
}
