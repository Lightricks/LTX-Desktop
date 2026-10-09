import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef } from "react";

import type { components } from "../../../generated/backend-openapi.ts";
import { unwrapApiResult } from "../../lib/unwrapApiResult.ts";
import { useExploreRuntime } from "../../runtime/ExploreRuntime.tsx";

export type DashboardSelection = Required<components["schemas"]["DashboardSelection"]>;
export type DashboardRange = DashboardSelection["range"];

const DASHBOARD_SELECTION_KEY = ["settings", "dashboard-selection"] as const;

const EMPTY_DASHBOARD_SELECTION: DashboardSelection = {
  range: "7d",
  models: [],
  resolutions: [],
  aspectRatios: [],
  fps: [],
};

function dashboardSelection(
  value: components["schemas"]["DashboardSelection"] | undefined,
): DashboardSelection {
  return {
    range: value?.range === "30d" || value?.range === "all" ? value.range : "7d",
    models: value?.models ?? [],
    resolutions: value?.resolutions ?? [],
    aspectRatios: value?.aspectRatios ?? [],
    fps: value?.fps ?? [],
  };
}

export function useDashboardSelection(): {
  selection: DashboardSelection;
  ready: boolean;
  setSelection: (patch: Partial<DashboardSelection>) => void;
} {
  const { api } = useExploreRuntime();
  const queryClient = useQueryClient();
  const saves = useRef<Promise<unknown>>(Promise.resolve());
  const query = useQuery({
    queryKey: DASHBOARD_SELECTION_KEY,
    queryFn: async () => {
      return dashboardSelection(unwrapApiResult(await api.getDashboardSelection()));
    },
  });
  const ready = query.isSuccess || query.isError;

  return {
    selection: query.data ?? EMPTY_DASHBOARD_SELECTION,
    ready,
    setSelection: (patch) => {
      if (!ready) return;
      const current = dashboardSelection(
        queryClient.getQueryData<DashboardSelection>(DASHBOARD_SELECTION_KEY),
      );
      const next = dashboardSelection({ ...current, ...patch });
      queryClient.setQueryData(DASHBOARD_SELECTION_KEY, next);
      saves.current = saves.current
        .then(async () => unwrapApiResult(await api.updateDashboardSelection(next)))
        .catch(() => {
          if (queryClient.getQueryData(DASHBOARD_SELECTION_KEY) === next) {
            queryClient.setQueryData(DASHBOARD_SELECTION_KEY, current);
          }
        });
    },
  };
}
