import { useQuery } from "@tanstack/react-query";

import { useExploreRuntime } from "../runtime/ExploreRuntime";
import {
  GENERATION_POLLING_QUERY_OPTIONS,
  generationRefetchInterval,
} from "../runtime/generationPollingPolicy";
import { useInvalidateAssetsOnSucceededSnapshot } from "./useInvalidateAssetsOnGenerationSuccess";
import {
  fetchGenerations,
  generationQueryKeys,
} from "./generationQueryKeys";

export function useGenerations(
  feature: string,
  options?: { keepPolling?: boolean },
) {
  const { api, generationPolling } = useExploreRuntime();
  const keepPolling = options?.keepPolling ?? false;

  const query = useQuery({
    queryKey: generationQueryKeys.list(feature),
    queryFn: () => fetchGenerations(api, feature),
    ...GENERATION_POLLING_QUERY_OPTIONS,
    refetchInterval: (query) =>
      generationRefetchInterval(
        generationPolling,
        {
          data: query.state.data,
          error: query.state.error,
          fetchFailureCount: query.state.fetchFailureCount,
        },
        keepPolling,
      ),
  });
  useInvalidateAssetsOnSucceededSnapshot(query.data);
  return query;
}
