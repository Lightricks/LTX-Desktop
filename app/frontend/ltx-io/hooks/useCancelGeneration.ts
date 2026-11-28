import { useMutation, useQueryClient } from "@tanstack/react-query";

import { unwrapApiResult } from "../lib/unwrapApiResult";
import { useExploreRuntime } from "../runtime/ExploreRuntime";

import { invalidateAffectedGenerationQueries } from "./generationQueryKeys";

export const GENERATION_CANCEL_CONFIRMATION = {
  confirmationName: "generation-cancel-confirm",
  title: "Are you sure you want to delete this generation?",
  description: "This stops the generation and removes it from the queue.",
  confirmLabel: "Delete generation",
} as const;

export function useCancelGeneration() {
  const { api } = useExploreRuntime();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (generationId: string) =>
      unwrapApiResult(await api.cancelQueuedGeneration(generationId)),
    onSuccess: () => {
      invalidateAffectedGenerationQueries(queryClient);
    },
  });
}
