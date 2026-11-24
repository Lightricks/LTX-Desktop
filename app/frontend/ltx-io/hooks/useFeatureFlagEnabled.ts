import type { DevFlagKey } from "@/lib/dev-flags";

import { useExploreRuntime } from "../runtime/ExploreRuntime.tsx";
import { useFeatureFlags } from "./useFeatureFlags.ts";

/**
 * One feature flag, on Desktop and on Remote. It is false until the backend answers, so a
 * feature behind the flag stays hidden until the flag is known.
 */
export function useFeatureFlagEnabled(flag: DevFlagKey): boolean {
  const { api } = useExploreRuntime();
  return useFeatureFlags(api).data?.[flag] === true;
}
