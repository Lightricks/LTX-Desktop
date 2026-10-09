import { useMutation, useQueryClient } from "@tanstack/react-query";

import {
  fetchQueueSnapshotAfterReorder,
  generationQueryKeys,
  replaceGenerationQueueCache,
  type ExploreQueueSnapshot,
  type GenerationQueueApi,
  type ReorderGenerationQueueRequest,
} from "./generationQueryKeys.ts";

export function applyOptimisticQueueReorder(
  snapshot: ExploreQueueSnapshot,
  request: ReorderGenerationQueueRequest,
): ExploreQueueSnapshot {
  const moved = snapshot.queued.find(
    (entry) => entry.generation.id === request.generation_id,
  );
  if (moved == null) {
    return snapshot;
  }

  const queued = snapshot.queued.filter(
    (entry) => entry.generation.id !== request.generation_id,
  );
  if (request.before_generation_id == null) {
    return { ...snapshot, queued: [...queued, moved] };
  }

  const targetIndex = queued.findIndex(
    (entry) => entry.generation.id === request.before_generation_id,
  );
  if (targetIndex < 0) {
    return { ...snapshot, queued: [...queued, moved] };
  }

  return {
    ...snapshot,
    queued: [
      ...queued.slice(0, targetIndex),
      moved,
      ...queued.slice(targetIndex),
    ],
  };
}

export function useReorderGenerationQueue(options: { api: GenerationQueueApi }) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: ReorderGenerationQueueRequest) =>
      fetchQueueSnapshotAfterReorder(options.api, request),
    onMutate: async (request) => {
      await queryClient.cancelQueries({ queryKey: generationQueryKeys.queue });
      const previous = queryClient.getQueryData<ExploreQueueSnapshot>(
        generationQueryKeys.queue,
      );
      if (previous != null) {
        replaceGenerationQueueCache(
          queryClient,
          applyOptimisticQueueReorder(previous, request),
        );
      }
      return { previous };
    },
    onError: (_error, _request, context) => {
      if (context?.previous != null) {
        replaceGenerationQueueCache(queryClient, context.previous);
      }
    },
    onSuccess: (snapshot) => {
      replaceGenerationQueueCache(queryClient, snapshot);
      void queryClient.invalidateQueries({ queryKey: generationQueryKeys.all });
    },
  });
}
