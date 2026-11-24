import { useCallback, useRef, useState } from "react";

import { type PromptHistoryEntry, shouldApplyEnhanceResult } from "@/lib/prompt-provenance";

import { useExploreRuntime } from "../runtime/ExploreRuntime";
import { enhanceFailureMessage, historyAfterEnhance } from "./manualEnhanceOutcome";
import { useEnhanceProviderChoice } from "./useEnhanceProviderChoice";

/**
 * The Enhance button for an image prompt. The button shows only while auto-enhance is off. With
 * auto-enhance on, the backend rewrites a typed prompt when the job runs.
 */
export function useImageEnhance({
  prompt,
  promptHistory,
  historyIndex,
  onPromptHistoryChange,
  disabled,
}: {
  prompt: string;
  promptHistory: PromptHistoryEntry[];
  historyIndex: number;
  onPromptHistoryChange: (history: PromptHistoryEntry[], index: number) => void;
  disabled: boolean;
}) {
  const { api, onGeminiKeyRequired } = useExploreRuntime();
  const choice = useEnhanceProviderChoice();
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [enhanceError, setEnhanceError] = useState<string | null>(null);
  const promptRef = useRef(prompt);
  promptRef.current = prompt;

  const hasPrompt = prompt.trim().length > 0;
  const providerReady =
    choice.enhanceProvider === "api" ? choice.hasGeminiApiKey : choice.hasLocalTextEncoder;
  const keyMissing = choice.enhanceProvider === "api" && !choice.hasGeminiApiKey;

  const runEnhance = useCallback(async () => {
    if (isEnhancing || !hasPrompt) return;
    setIsEnhancing(true);
    setEnhanceError(null);
    const sourcePrompt = prompt;
    try {
      const result = await api.enhancePrompt({
        prompt: sourcePrompt,
        loraCatalogIds: [],
        provider: choice.enhanceProvider,
        mediaType: "image",
      });
      if (!result.ok) {
        setEnhanceError(enhanceFailureMessage(result.error, onGeminiKeyRequired));
        return;
      }
      if (!shouldApplyEnhanceResult(promptRef.current, sourcePrompt)) return;
      const next = historyAfterEnhance({
        history: promptHistory,
        index: historyIndex,
        sourcePrompt,
        enhancedPrompt: result.data.enhancedPrompt,
      });
      onPromptHistoryChange(next.history, next.index);
    } catch (error) {
      setEnhanceError(error instanceof Error ? error.message : "Enhance failed.");
    } finally {
      setIsEnhancing(false);
    }
  }, [
    api,
    choice.enhanceProvider,
    hasPrompt,
    historyIndex,
    isEnhancing,
    onGeminiKeyRequired,
    onPromptHistoryChange,
    prompt,
    promptHistory,
  ]);

  return {
    showManualEnhance: choice.showManualEnhance,
    canEnhance:
      choice.showManualEnhance &&
      hasPrompt &&
      !disabled &&
      !isEnhancing &&
      (providerReady || (keyMissing && onGeminiKeyRequired !== undefined)),
    isEnhancing,
    enhanceError,
    runEnhance,
    enhanceProvider: choice.enhanceProvider,
    canToggleEnhanceProvider: choice.canToggleEnhanceProvider,
    setEnhanceProvider: choice.setEnhanceProvider,
  };
}
