import { useEffect, useRef } from "react";
import { useQueries, useQueryClient } from "@tanstack/react-query";

import { HOME_FEATURES } from "@/lib/home-features";

import {
  hasNewSucceededGeneration,
  type Generation,
} from "../lib/resultsFeedModel";
import { useExploreRuntime } from "../runtime/ExploreRuntime";
import {
  GENERATION_POLLING_QUERY_OPTIONS,
  generationRefetchInterval,
  type GenerationPollQueryState,
} from "../runtime/generationPollingPolicy";
import { assetQueryKeys } from "./assetQueryKeys";
import { fetchGenerations, generationQueryKeys } from "./generationQueryKeys";

const GENERATION_FEATURES = HOME_FEATURES.map((feature) => feature.id);

export function useInvalidateAssetsOnSucceededSnapshot(
  data: Generation[] | undefined,
) {
  const queryClient = useQueryClient();
  const previousRef = useRef<Generation[] | undefined>(undefined);

  useEffect(() => {
    if (hasNewSucceededGeneration(previousRef.current, data)) {
      void queryClient.invalidateQueries({ queryKey: assetQueryKeys.all });
    }
    previousRef.current = data;
  }, [data, queryClient]);
}

/** Poll Explore features while jobs are in flight; refresh Assets on succeeded. */
export function useInvalidateAssetsOnGenerationSuccess() {
  const { api, generationPolling } = useExploreRuntime();
  const queryClient = useQueryClient();
  const previousByFeatureRef = useRef(new Map<string, Generation[]>());

  const queries = useQueries({
    queries: GENERATION_FEATURES.map((feature) => ({
      queryKey: generationQueryKeys.list(feature),
      queryFn: () => fetchGenerations(api, feature),
      ...GENERATION_POLLING_QUERY_OPTIONS,
      refetchInterval: (query: { state: GenerationPollQueryState }) =>
        generationRefetchInterval(
          generationPolling,
          {
            data: query.state.data,
            error: query.state.error,
            fetchFailureCount: query.state.fetchFailureCount,
          },
          false,
        ),
    })),
  });

  const succeededKey = queries
    .map((query, index) => {
      const feature = GENERATION_FEATURES[index] ?? "";
      const succeeded = (query.data ?? [])
        .filter((generation) => generation.status === "succeeded")
        .map((generation) => generation.id)
        .join(",");
      return `${feature}:${succeeded}`;
    })
    .join("|");

  useEffect(() => {
    let shouldInvalidate = false;
    for (const [index, feature] of GENERATION_FEATURES.entries()) {
      const next = queries[index]?.data;
      const previous = previousByFeatureRef.current.get(feature);
      if (hasNewSucceededGeneration(previous, next)) {
        shouldInvalidate = true;
      }
      if (next !== undefined) {
        previousByFeatureRef.current.set(feature, next);
      }
    }
    if (shouldInvalidate) {
      void queryClient.invalidateQueries({ queryKey: assetQueryKeys.all });
    }
  }, [queryClient, queries, succeededKey]);
}
