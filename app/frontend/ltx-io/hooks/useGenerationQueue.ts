import { useQuery } from "@tanstack/react-query";

import {
  GENERATION_POLLING_QUERY_OPTIONS,
  queueRefetchInterval,
  type ExploreGenerationPollingPolicy,
  type QueuePollQueryState,
} from "../runtime/generationPollingPolicy.ts";
import {
  fetchGenerationQueue,
  generationQueryKeys,
  type GenerationQueueApi,
} from "./generationQueryKeys.ts";

export type UseGenerationQueueOptions = {
  api: GenerationQueueApi;
  generationPolling: ExploreGenerationPollingPolicy;
  isLive: boolean;
};

export function generationQueueQueryOptions({
  api,
  generationPolling,
  isLive,
}: UseGenerationQueueOptions) {
  return {
    queryKey: generationQueryKeys.queue,
    queryFn: () => fetchGenerationQueue(api),
    enabled: true,
    ...GENERATION_POLLING_QUERY_OPTIONS,
    refetchInterval: (query: { state: QueuePollQueryState }) =>
      queueRefetchInterval(
        generationPolling,
        {
          data: query.state.data,
          error: query.state.error,
          fetchFailureCount: query.state.fetchFailureCount,
        },
        isLive,
      ),
  };
}

export function useGenerationQueue(options: UseGenerationQueueOptions) {
  return useQuery(generationQueueQueryOptions(options));
}
