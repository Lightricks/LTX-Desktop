import { useEffect, useState } from "react";

import type { ApiSuccessOf } from "@/lib/api-client";

import { useExploreRuntime } from "../runtime/ExploreRuntime";

export type PromptEnhancerStatus = ApiSuccessOf<"getPromptEnhancer">;

/** The backend's enhancer status. Null until it loads. Models changing refetches it. */
export function usePromptEnhancerStatus(): PromptEnhancerStatus | null {
  const { api, modelsVersion } = useExploreRuntime();
  const [status, setStatus] = useState<PromptEnhancerStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    void api.getPromptEnhancer().then((result) => {
      if (!cancelled && result.ok) {
        setStatus(result.data);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [api, modelsVersion]);

  return status;
}
