import { useExploreRuntime } from "../runtime/ExploreRuntime";
import { usePromptEnhancerStatus } from "./usePromptEnhancerStatus";

/**
 * Why the negative prompt (NAG) is off for a feature, or null when it can run. The backend
 * status answers per feature. Until it loads, the runtime's reason (Desktop Settings) is used.
 */
export function useNagUnavailableReason(featureId: string): string | null {
  const { nagUnavailableReason: runtimeNagUnavailableReason } = useExploreRuntime();
  const status = usePromptEnhancerStatus();
  return status?.nagUnavailableReasonByFeature !== undefined
    ? (status.nagUnavailableReasonByFeature[featureId] ?? null)
    : (status?.nagUnavailableReason ?? runtimeNagUnavailableReason ?? null);
}
