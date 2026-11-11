import { useQuery } from "@tanstack/react-query";

import { unwrapApiResult } from "../lib/unwrapApiResult";
import { useExploreRuntime } from "../runtime/ExploreRuntime";

/**
 * The "Show advanced controls" setting. Desktop passes it in the runtime, so a toggle in
 * Settings shows at once. Remote has no settings screen: it reads the host's value and
 * refreshes it when the page regains focus. Off while unknown.
 */
export function useAdvancedControls(): boolean {
  const { api, advancedControls } = useExploreRuntime();
  const query = useQuery({
    queryKey: ["ui-preferences"],
    queryFn: async () => unwrapApiResult(await api.getUiPreferences()),
    enabled: advancedControls === undefined,
    // The app-wide client turns focus refetch off. Remote needs it to see a Desktop change.
    refetchOnWindowFocus: true,
    retry: false,
  });
  return advancedControls ?? query.data?.advancedControls ?? false;
}
