import { useMutation, useQueryClient } from "@tanstack/react-query";

import { unwrapApiResult } from "../lib/unwrapApiResult";
import { useExploreRuntime } from "../runtime/ExploreRuntime";
import {
  generationQueryKeys,
  invalidateAffectedGenerationQueries,
} from "./generationQueryKeys";

export function useDeleteGeneration(feature: string) {
  const { api } = useExploreRuntime();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (generationId: string) =>
      unwrapApiResult(await api.deleteGeneration(generationId)),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: generationQueryKeys.list(feature),
      });
      invalidateAffectedGenerationQueries(queryClient);
    },
  });
}
