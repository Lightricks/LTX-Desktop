import { useQueries } from "@tanstack/react-query";
import { useMemo } from "react";

import type { Generation } from "../lib/resultsFeedModel";
import { isInFlightGeneration } from "../lib/resultsFeedModel";
import { unwrapApiResult } from "../lib/unwrapApiResult";
import { useExploreRuntime } from "../runtime/ExploreRuntime";
import { GENERATION_POLLING_QUERY_OPTIONS } from "../runtime/generationPollingPolicy";
import { generationQueryKeys } from "./generationQueryKeys";
import { useInvalidateAssetsOnSucceededSnapshot } from "./useInvalidateAssetsOnGenerationSuccess";

/** Generations by id, polled at the queue rate while they are in flight. A null id has none. */
export function useGenerationDetails(
  generationIds: readonly (string | null)[],
): (Generation | undefined)[] {
  const { api, generationPolling } = useExploreRuntime();
  const results = useQueries({
    queries: generationIds.map((generationId) => ({
      queryKey: generationQueryKeys.detail(generationId ?? ""),
      enabled: generationId !== null,
      queryFn: async () => unwrapApiResult(await api.getGeneration(generationId as string)),
      ...GENERATION_POLLING_QUERY_OPTIONS,
      refetchInterval: (current: {
        state: { error: unknown; fetchFailureCount: number; data: Generation | undefined };
      }) => {
        if (current.state.error) {
          return generationPolling.errorBackoffMs(current.state.fetchFailureCount);
        }
        const data = current.state.data;
        return data && !isInFlightGeneration(data)
          ? false
          : generationPolling.queueActiveIntervalMs;
      },
    })),
  });
  const generations = results.map((result) => result.data);
  const key = generations.map((generation) => `${generation?.id}:${generation?.status}`).join("|");
  const snapshot = useMemo(
    () => generations.filter((generation) => generation !== undefined),
    // The key changes exactly when an id or a status does.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [key],
  );
  useInvalidateAssetsOnSucceededSnapshot(snapshot);
  return generations;
}

export function useGenerationDetail(generationId: string | null): Generation | undefined {
  return useGenerationDetails([generationId])[0];
}
