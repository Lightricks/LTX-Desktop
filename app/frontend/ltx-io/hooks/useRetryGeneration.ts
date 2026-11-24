import { useMutation, useQueryClient } from "@tanstack/react-query";

import { unwrapApiResult } from "../lib/unwrapApiResult";
import { useExploreRuntime } from "../runtime/ExploreRuntime";

import { assetQueryKeys } from "./assetQueryKeys";
import { invalidateAffectedGenerationQueries } from "./generationQueryKeys";

export function useRetryGeneration() {
  const { api } = useExploreRuntime();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (generationId: string) =>
      unwrapApiResult(await api.retryGeneration(generationId)),
    onSuccess: () => {
      invalidateAffectedGenerationQueries(queryClient);
      void queryClient.invalidateQueries({ queryKey: assetQueryKeys.all });
    },
  });
}
