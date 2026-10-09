import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import { generationQueryKeys } from "../../hooks/generationQueryKeys.ts";
import { useGenerationQueue } from "../../hooks/useGenerationQueue.ts";
import { useExploreRuntime } from "../../runtime/ExploreRuntime.tsx";

/** Watches the queue at the slow background rate (the queue panel speeds it up when opened), and reloads the dashboard when a run ends. */
export function useRefreshDashboardOnFinish(): void {
  const queryClient = useQueryClient();
  const { api, generationPolling } = useExploreRuntime();
  const { data } = useGenerationQueue({ api, generationPolling, isLive: false });
  const finished = data
    ? [...data.done, ...data.failed].map((entry) => entry.generation.id).join(",")
    : null;
  const previous = useRef(finished);

  useEffect(() => {
    if (previous.current !== null && finished !== null && finished !== previous.current) {
      void queryClient.invalidateQueries({ queryKey: generationQueryKeys.dashboards });
    }
    previous.current = finished;
  }, [finished, queryClient]);
}
