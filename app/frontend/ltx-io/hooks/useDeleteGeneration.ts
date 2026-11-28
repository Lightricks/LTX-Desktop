import { useMutation, useQueryClient } from "@tanstack/react-query";

import { unwrapApiResult } from "../lib/unwrapApiResult";
import { useExploreRuntime } from "../runtime/ExploreRuntime";

import { invalidateAffectedGenerationQueries } from "./generationQueryKeys";

export function useDeleteGeneration() {
  const { api } = useExploreRuntime();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (generationId: string) =>
      unwrapApiResult(await api.deleteGeneration(generationId)),
    onSuccess: () => {
      invalidateAffectedGenerationQueries(queryClient);
    },
  });
}
