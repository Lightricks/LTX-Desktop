import { useQuery } from "@tanstack/react-query";
import { useDeferredValue, useMemo } from "react";

import type { components } from "@/generated/backend-openapi";
import { useExploreRuntime } from "../runtime/ExploreRuntime";
import { unwrapApiResult } from "../lib/unwrapApiResult";
import type {
  EditPreflightFeature,
  EditPreflightState,
} from "../lib/editPreflight";
import type { EditContextMode } from "../lib/editContextMode";

type EditPreflightCandidate =
  components["schemas"]["EditPreflightCandidate"];

export type EditPreflightRequestInput = {
  feature: EditPreflightFeature;
  assetId: string | null;
  startTime: number;
  mode: "start" | "end";
  /** How much of the clip the edit keeps as context. Sets the window it is priced at. */
  contextMode: EditContextMode;
  candidates: EditPreflightCandidate[];
  durations: number[];
};

/**
 * Advisory edit verdicts for the current selection. Deferred (trim drags
 * settle before refetching) and sticky (previous verdicts stay up while the
 * next request flies, so pickers never flash enabled mid-drag). Null while
 * pending, errored, or disabled — unknown always means allowed.
 */
export function useEditPreflight(
  input: EditPreflightRequestInput,
): EditPreflightState | null {
  const { api, budgetMode } = useExploreRuntime();
  const deferred = useDeferredValue(input);
  const queryKey = useMemo(
    () => [
      "edit-preflight",
      budgetMode ?? "server",
      deferred.feature,
      deferred.assetId,
      deferred.startTime,
      deferred.mode,
      deferred.contextMode,
      deferred.candidates
        .map((candidate) => `${candidate.key}:${candidate.width}x${candidate.height}`)
        .join(","),
      deferred.durations.join(","),
    ],
    [budgetMode, deferred],
  );
  const enabled =
    deferred.assetId != null &&
    deferred.candidates.length > 0 &&
    deferred.durations.length > 0;
  const query = useQuery({
    queryKey,
    queryFn: async (): Promise<EditPreflightState> => {
      const assetId = deferred.assetId as string;
      const response = unwrapApiResult(
        await api.preflightEdit({
          feature: deferred.feature,
          assetId,
          startTime: deferred.startTime,
          mode: deferred.mode,
          contextMode: deferred.contextMode,
          candidates: deferred.candidates,
          durations: deferred.durations,
        }),
      );
      return { assetId, contextMode: deferred.contextMode, response };
    },
    enabled,
    retry: false,
    staleTime: 30_000,
    placeholderData: (previous) => {
      if (budgetMode == null) return previous;
      return previous?.response.budgetMode === budgetMode ? previous : undefined;
    },
  });
  if (!enabled) return null;
  return query.data ?? null;
}
