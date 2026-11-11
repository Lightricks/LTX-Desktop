import { useMemo } from "react";

import type { HomeFeatureFlags } from "../../lib/home-features.ts";
import { useFeatureFlagEnabled } from "./useFeatureFlagEnabled.ts";

/**
 * The flags that hide a Home feature. Each is false until the backend answers, so a
 * feature behind a flag stays hidden until the flag is known.
 */
export function useHomeFeatureFlags(): HomeFeatureFlags {
  const keyframes = useFeatureFlagEnabled("keyframes");
  return useMemo(() => ({ keyframes }), [keyframes]);
}
