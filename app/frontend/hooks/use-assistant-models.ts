import { useQuery } from "@tanstack/react-query";

import { ApiClient } from "../lib/api-client";

export const assistantModelsQueryKey = ["assistant-models"] as const;

export function useAssistantModels(
  active = true,
  listModels: typeof ApiClient.listAssistantModels = ApiClient.listAssistantModels,
) {
  return useQuery({
    queryKey: assistantModelsQueryKey,
    enabled: active,
    queryFn: async () => {
      const result = await listModels();
      if (!result.ok) throw new Error(result.error.message);
      return result.data;
    },
    refetchInterval: (query) =>
      query.state.data?.download?.status === "downloading" ? 1000 : 5000,
    retry: false,
  });
}
