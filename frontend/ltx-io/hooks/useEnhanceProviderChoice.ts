import { useCallback, useState } from "react";

import type { EnhanceProvider } from "@/hooks/use-prompt-enhancer-provider";
import { resolveUsableEnhanceProvider } from "@/lib/enhance-provider";

import { useExploreRuntime } from "../runtime/ExploreRuntime";
import { resolveShowManualEnhance } from "./manualEnhanceVisibility";
import { usePromptEnhancerStatus } from "./usePromptEnhancerStatus";

/**
 * Which Enhance provider runs, which are choosable, and whether the Enhance button shows.
 * `status` is the backend status, null until it loads.
 */
export function useEnhanceProviderChoice() {
  const { persistEnhanceProviderPreference, readEnhanceProviderPreference, autoEnhancePrompts } =
    useExploreRuntime();
  const status = usePromptEnhancerStatus();
  const [providerOverride, setProviderOverride] = useState<EnhanceProvider | null>(
    () => readEnhanceProviderPreference?.() ?? null,
  );

  const hasGeminiApiKey = status?.hasGeminiApiKey ?? false;
  const hasLocalTextEncoder = status?.localEnhancementSupported ?? false;
  const enhanceProvider = resolveUsableEnhanceProvider({
    preference: providerOverride,
    hasGeminiApiKey,
    hasLocalTextEncoder,
    fallback: status?.defaultProvider ?? "api",
  });
  const showManualEnhance = resolveShowManualEnhance(
    autoEnhancePrompts,
    status?.showManualEnhance,
  );

  const setEnhanceProvider = useCallback(
    (provider: EnhanceProvider) => {
      setProviderOverride(provider);
      persistEnhanceProviderPreference?.(provider);
    },
    [persistEnhanceProviderPreference],
  );

  return {
    hasGeminiApiKey,
    hasLocalTextEncoder,
    enhanceProvider,
    canToggleEnhanceProvider: status?.canToggleProvider ?? false,
    showManualEnhance,
    setEnhanceProvider,
    status,
  };
}
