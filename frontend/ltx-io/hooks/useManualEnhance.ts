import { useCallback, useEffect, useRef, useState } from "react";

import type { EnhanceProvider } from "@/hooks/use-prompt-enhancer-provider";
import { isEnhanceBlockedByMissingGeminiKey } from "@/lib/enhance-gemini-key";
import { resolveUsableEnhanceProvider } from "@/lib/enhance-provider";
import type { HomeFeatureId } from "@/lib/home-features";
import { getIcLoraRecipe, isIcLoraRecipeId } from "@/lib/ic-lora-recipes";
import { getLoraRecipe, isLoraRecipeId } from "@/lib/lora-recipes";
import type { PromptProvenance } from "@/lib/prompt-provenance";
import {
  type PromptHistoryEntry,
  appendEnhanceResult,
  resolvePromptProvenance,
  shouldApplyEnhanceResult,
} from "@/lib/prompt-provenance";

import { useExploreRuntime } from "../runtime/ExploreRuntime";
import { useFeatureFormStore } from "../stores/featureFormStore";
import { resolveShowManualEnhance } from "./manualEnhanceVisibility";

type PromptValues = {
  prompt?: string;
  startFrame?: { assetId: string } | null;
  endFrame?: { assetId: string } | null;
};

type PromptEnhancerStatus = {
  exploreAutoEnhancePrompts: boolean;
  hasGeminiApiKey: boolean;
  localEnhancementSupported: boolean;
  defaultProvider: EnhanceProvider;
  canToggleProvider: boolean;
  showManualEnhance: boolean;
};

const MISSING_GEMINI_KEY_MESSAGE =
  "Add a Gemini API key in Desktop Settings, or use local Enhance when Gemma is downloaded.";

function liveFormState(featureId: HomeFeatureId): {
  prompt: string | undefined;
  history: PromptHistoryEntry[] | undefined;
  index: number | undefined;
} {
  const state = useFeatureFormStore.getState();
  const values = state.getValues(featureId);
  const stored = state.promptHistoryByFeature[featureId];
  return {
    prompt:
      values != null && "prompt" in values && typeof values.prompt === "string"
        ? values.prompt
        : undefined,
    history: stored?.history,
    index: stored?.index,
  };
}

export function useManualEnhance({
  featureId,
  values,
  promptHistory,
  historyIndex,
  onPromptHistoryChange,
  isCreating,
}: {
  featureId: HomeFeatureId;
  values: PromptValues;
  promptHistory: PromptHistoryEntry[];
  historyIndex: number;
  onPromptHistoryChange: (history: PromptHistoryEntry[], index: number) => void;
  isCreating: boolean;
}) {
  const {
    api,
    modelsVersion,
    persistEnhanceProviderPreference,
    readEnhanceProviderPreference,
    autoEnhancePrompts,
    onGeminiKeyRequired,
  } = useExploreRuntime();
  const [status, setStatus] = useState<PromptEnhancerStatus | null>(null);
  const [providerOverride, setProviderOverride] = useState<EnhanceProvider | null>(
    () => readEnhanceProviderPreference?.() ?? null,
  );

  useEffect(() => {
    let cancelled = false;
    void api.getPromptEnhancer().then((result) => {
      if (!cancelled && result.ok) {
        setStatus(result.data);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [api, modelsVersion]);

  const hasGeminiApiKey = status?.hasGeminiApiKey ?? false;
  const hasLocalTextEncoder = status?.localEnhancementSupported ?? false;
  const enhanceProvider = resolveUsableEnhanceProvider({
    preference: providerOverride,
    hasGeminiApiKey,
    hasLocalTextEncoder,
    fallback: status?.defaultProvider ?? "api",
  });
  const canToggleEnhanceProvider = status?.canToggleProvider ?? false;
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

  const [isEnhancing, setIsEnhancing] = useState(false);
  const [enhanceError, setEnhanceError] = useState<string | null>(null);

  const prompt = String(values.prompt ?? "");
  const hasPrompt = prompt.trim().length > 0;
  const hasStartFrame = Boolean(values.startFrame?.assetId);
  const hasEnhanceInput = hasPrompt || hasStartFrame;
  const promptRef = useRef(prompt);
  promptRef.current = prompt;

  const runEnhance = useCallback(async () => {
    if (isEnhancing || !hasEnhanceInput) return;
    setIsEnhancing(true);
    setEnhanceError(null);

    const sourcePrompt = prompt;
    const loraCatalogIds: string[] = [];
    const imageAssetId =
      featureId === "image-to-video" || featureId === "audio-to-video"
        ? values.startFrame?.assetId
        : undefined;
    const lastImageAssetId =
      featureId === "image-to-video" ? values.endFrame?.assetId : undefined;
    if (isLoraRecipeId(featureId)) {
      loraCatalogIds.push(getLoraRecipe(featureId).catalogId);
    }
    const icLoraId = isIcLoraRecipeId(featureId)
      ? getIcLoraRecipe(featureId).catalogId
      : undefined;

    try {
      const result = await api.enhancePrompt({
        prompt: sourcePrompt,
        loraCatalogIds,
        icLoraId,
        imageAssetId,
        lastImageAssetId,
        provider: enhanceProvider,
        mediaType: "video",
      });

      if (!result.ok) {
        if (
          result.error.code === "GEMINI_INVALID_API_KEY" ||
          result.error.code === "GEMINI_API_KEY_MISSING"
        ) {
          onGeminiKeyRequired?.();
          setEnhanceError(
            onGeminiKeyRequired ? result.error.message : MISSING_GEMINI_KEY_MESSAGE,
          );
        } else {
          setEnhanceError(result.error.message);
        }
        return;
      }

      const live = liveFormState(featureId);
      const history = live.history ?? promptHistory;
      const index = live.index ?? historyIndex;
      if (
        !shouldApplyEnhanceResult(live.prompt ?? promptRef.current, sourcePrompt)
      ) {
        return;
      }

      const next = appendEnhanceResult({
        history,
        index,
        sourcePrompt,
        sourceProvenance: resolvePromptProvenance(
          sourcePrompt,
          history,
          index,
        ),
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
    enhanceProvider,
    featureId,
    hasEnhanceInput,
    historyIndex,
    isEnhancing,
    onGeminiKeyRequired,
    onPromptHistoryChange,
    prompt,
    promptHistory,
    values.endFrame?.assetId,
    values.startFrame?.assetId,
  ]);

  const canEnhance =
    showManualEnhance &&
    hasEnhanceInput &&
    !isCreating &&
    !isEnhancing &&
    (enhanceProvider === "api" ? hasGeminiApiKey : hasLocalTextEncoder);

  const enhanceBlockedByMissingGeminiKey = isEnhanceBlockedByMissingGeminiKey({
    enhanceAvailableForMode: showManualEnhance,
    enhanceProvider,
    hasGeminiApiKey,
    hasEnhanceInput,
    isGenerationInProgressForEnhance: isCreating || isEnhancing,
    isOtherGenerationRunning: false,
  });

  const promptProvenance: PromptProvenance = resolvePromptProvenance(
    prompt,
    promptHistory,
    historyIndex,
  );

  return {
    showManualEnhance,
    canEnhance:
      canEnhance || Boolean(enhanceBlockedByMissingGeminiKey && onGeminiKeyRequired),
    enhanceBlockedByMissingGeminiKey,
    isEnhancing,
    enhanceError,
    runEnhance,
    promptProvenance,
    enhanceProvider,
    canToggleEnhanceProvider,
    setEnhanceProvider,
  };
}
