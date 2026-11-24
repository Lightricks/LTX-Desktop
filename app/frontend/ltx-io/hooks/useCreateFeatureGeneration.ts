import { useMutation, useQueryClient } from "@tanstack/react-query";

import { unwrapApiResult, type ApiResult } from "../lib/unwrapApiResult";
import { useExploreRuntime } from "../runtime/ExploreRuntime";
import { assetQueryKeys } from "./assetQueryKeys";
import { invalidateAffectedGenerationQueries } from "./generationQueryKeys";

export function useCreateFeatureGeneration<TBody, TData>(
  create: (
    api: ReturnType<typeof useExploreRuntime>["api"],
    body: TBody,
  ) => Promise<ApiResult<TData>>,
) {
  const { api } = useExploreRuntime();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (body: TBody) => unwrapApiResult(await create(api, body)),
    onSuccess: () => {
      invalidateAffectedGenerationQueries(queryClient);
      void queryClient.invalidateQueries({ queryKey: assetQueryKeys.all });
    },
  });
}
