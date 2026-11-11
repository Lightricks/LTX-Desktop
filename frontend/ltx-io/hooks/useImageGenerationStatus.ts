import { useQuery } from "@tanstack/react-query";

import { unwrapApiResult } from "../lib/unwrapApiResult";
import { useExploreRuntime } from "../runtime/ExploreRuntime";

/** Which image providers can run now. A model download or a key change refetches it. */
export function useImageGenerationStatus() {
  const { api, modelsVersion } = useExploreRuntime();
  return useQuery({
    queryKey: ["image-generation-status", modelsVersion],
    queryFn: async () => unwrapApiResult(await api.getImageGenerationStatus()),
    refetchOnMount: true,
    refetchOnWindowFocus: true,
  });
}
