import { useQueryClient, type QueryKey } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import { useExploreRuntime } from "../runtime/ExploreRuntime.tsx";
import { useGenerationQueue } from "./useGenerationQueue.ts";

/** Watches the queue at the slow background rate (the queue panel speeds it up when opened), and invalidates `queryKey` when a run ends. Pass a stable key. */
export function useInvalidateOnGenerationFinish(queryKey: QueryKey): void {
  const queryClient = useQueryClient();
  const { api, generationPolling } = useExploreRuntime();
  const { data } = useGenerationQueue({ api, generationPolling, isLive: false });
  const finished = data
    ? [...data.done, ...data.failed].map((entry) => entry.generation.id).join(",")
    : null;
  const previous = useRef(finished);

  useEffect(() => {
    if (previous.current !== null && finished !== null && finished !== previous.current) {
      void queryClient.invalidateQueries({ queryKey });
    }
    previous.current = finished;
  }, [finished, queryClient, queryKey]);
}
