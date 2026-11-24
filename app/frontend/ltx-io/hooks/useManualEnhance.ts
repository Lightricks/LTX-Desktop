import { useCallback, useRef, useState } from "react";

import type { components } from "@/generated/backend-openapi";
import { CUSTOM_LORA_HOME_FEATURE } from "@/lib/custom-lora";
import { isEnhanceBlockedByMissingGeminiKey } from "@/lib/enhance-gemini-key";
import type { HomeFeatureId } from "@/lib/home-features";
import {
  icLoraCatalogId,
  isIcLoraRecipeId,
  isIcLoraSheetRecipeId,
} from "@/lib/ic-lora-recipes";
import { getLoraRecipe, isLoraRecipeId } from "@/lib/lora-recipes";
import type { PromptProvenance } from "@/lib/prompt-provenance";
import {
  type PromptHistoryEntry,
  resolvePromptProvenance,
  shouldApplyEnhanceResult,
} from "@/lib/prompt-provenance";

import { useExploreRuntime } from "../runtime/ExploreRuntime";
import { type FeatureFormKey, useFeatureFormStore } from "../stores/featureFormStore";
import { enhanceFailureMessage, historyAfterEnhance } from "./manualEnhanceOutcome";
import { useEnhanceProviderChoice } from "./useEnhanceProviderChoice";
import { useNagUnavailableReason } from "./useNagUnavailableReason";
import {
  type EnhanceKeyframe,
  assetIdOf,
  enhanceImageAssetId,
  enhanceKeyframeFields,
  hasManualEnhanceInput,
  isEnhanceSourceCurrent,
  keyframesSignature,
  lookImageAssetId,
  panelImageAssetIds,
  replacedWindowFor,
  resolveShowNagStrengthControls,
} from "./manualEnhanceVisibility";

type EnhancePromptRequest = components["schemas"]["EnhancePromptRequest"];

export type PromptValues = {
  prompt?: string;
  negativePrompt?: string;
  startFrame?: { assetId: string } | null;
  endFrame?: { assetId: string } | null;
  // The source video of an IC-LoRA recipe, Extend, or Retake.
  video?: { assetId: string } | null;
  // The look image of an IC-LoRA recipe that takes one (Layout to Render, Restore).
  image?: { assetId: string } | null;
  // The images of an image-list recipe (Ingredients), in order.
  images?: readonly { assetId: string }[] | null;
  // The stills of a Keyframes clip. `duration` (seconds) and `fps` are then the clip's.
  keyframes?: readonly EnhanceKeyframe[] | null;
  fps?: number;
  aspectRatio?: EnhancePromptRequest["aspectRatio"];
  resolution?: EnhancePromptRequest["resolution"];
  // The Retake selection, in seconds.
  startTime?: number;
  duration?: number;
};

function numberOrUndefined(value: unknown): number | undefined {
  return typeof value === "number" ? value : undefined;
}

function liveFormState(
  featureId: HomeFeatureId,
  storeKey: FeatureFormKey = featureId,
): {
  prompt: string | undefined;
  negativePrompt: string | undefined;
  history: PromptHistoryEntry[] | undefined;
  index: number | undefined;
  // null: the form has no source video. undefined: no live values to compare.
  videoAssetId: string | null | undefined;
  // The live start frame of Image-to-Video, Audio-to-Video, or Motion Track.
  startFrameAssetId: string | null | undefined;
  // The live look image of an IC-LoRA recipe. Same null and undefined meaning as the video.
  referenceImageAssetId: string | null | undefined;
  // The live panel images of an image-list recipe, in order. Empty for any other feature.
  panelAssetIds: string[] | undefined;
  // The live stills of a Keyframes clip, as a `keyframesSignature`. Empty for any other feature.
  keyframes: string | undefined;
  // The live Retake selection. undefined for any other feature, or when the values are gone.
  replacedWindow: { start: number; end: number } | undefined;
} {
  const state = useFeatureFormStore.getState();
  const values = state.getValues(storeKey);
  const stored = state.promptHistoryByFeature[storeKey];
  return {
    prompt:
      values != null && "prompt" in values && typeof values.prompt === "string"
        ? values.prompt
        : undefined,
    negativePrompt:
      values != null && "negativePrompt" in values && typeof values.negativePrompt === "string"
        ? values.negativePrompt
        : undefined,
    history: stored?.history,
    index: stored?.index,
    videoAssetId: values == null ? undefined : assetIdOf("video" in values ? values.video : null),
    startFrameAssetId:
      values == null ? undefined : assetIdOf("startFrame" in values ? values.startFrame : null),
    referenceImageAssetId:
      values == null ? undefined : assetIdOf("image" in values ? values.image : null),
    panelAssetIds:
      values == null
        ? undefined
        : panelImageAssetIds({
            isSheetRecipe: isIcLoraSheetRecipeId(featureId),
            images: "images" in values && Array.isArray(values.images) ? values.images : null,
          }) ?? [],
    keyframes:
      values == null
        ? undefined
        : keyframesSignature(
            "keyframes" in values && Array.isArray(values.keyframes) ? values.keyframes : null,
            {
              duration: "duration" in values ? values.duration : undefined,
              fps: "fps" in values ? values.fps : undefined,
            },
          ),
    replacedWindow:
      values == null
        ? undefined
        : replacedWindowFor(featureId, {
            startTime: "startTime" in values ? numberOrUndefined(values.startTime) : undefined,
            duration: "duration" in values ? numberOrUndefined(values.duration) : undefined,
          }),
  };
}

export function useManualEnhance({
  featureId,
  storeKey,
  customLoraId,
  values,
  promptHistory,
  historyIndex,
  onPromptHistoryChange,
  isCreating,
}: {
  featureId: HomeFeatureId;
  storeKey?: FeatureFormKey;
  customLoraId?: string;
  values: PromptValues;
  promptHistory: PromptHistoryEntry[];
  historyIndex: number;
  onPromptHistoryChange: (history: PromptHistoryEntry[], index: number) => void;
  isCreating: boolean;
}) {
  const {
    api,
    onGeminiKeyRequired,
    showNagStrengthControls: runtimeShowNagStrengthControls,
  } = useExploreRuntime();
  const {
    hasGeminiApiKey,
    hasLocalTextEncoder,
    enhanceProvider,
    canToggleEnhanceProvider,
    showManualEnhance,
    setEnhanceProvider,
    status,
  } = useEnhanceProviderChoice();

  const [isEnhancing, setIsEnhancing] = useState(false);
  const [enhanceError, setEnhanceError] = useState<string | null>(null);

  const prompt = String(values.prompt ?? "");
  const negativePrompt = String(values.negativePrompt ?? "");
  const hasEnhanceInput = hasManualEnhanceInput({
    prompt,
    startFrame: values.startFrame,
    video: values.video,
    images: values.images,
    keyframes: values.keyframes,
  });
  const promptRef = useRef(prompt);
  promptRef.current = prompt;
  const negativePromptRef = useRef(negativePrompt);
  negativePromptRef.current = negativePrompt;

  const nagUnavailableReason = useNagUnavailableReason(featureId);
  const showNagStrengthControls = resolveShowNagStrengthControls(
    runtimeShowNagStrengthControls,
    status?.showNagStrengthControls,
  );

  const runEnhance = useCallback(async () => {
    if (isEnhancing || !hasEnhanceInput) return;
    setIsEnhancing(true);
    setEnhanceError(null);

    const sourcePrompt = prompt;
    const sourceNegativePrompt = nagUnavailableReason == null ? negativePrompt : "";
    const loraCatalogIds: string[] = [];
    const customLoraIds: string[] = [];
    const startFrameAssetId = values.startFrame?.assetId ?? null;
    const referenceImageAssetId = lookImageAssetId({
      isIcLoraRecipe: isIcLoraRecipeId(featureId),
      video: values.video,
      image: values.image,
    });
    const imageAssetId = enhanceImageAssetId({
      featureId,
      isIcLoraRecipe: isIcLoraRecipeId(featureId),
      startFrame: values.startFrame,
      video: values.video,
      image: values.image,
    });
    const lastImageAssetId =
      featureId === "image-to-video" ? values.endFrame?.assetId : undefined;
    const panelAssetIds = panelImageAssetIds({
      isSheetRecipe: isIcLoraSheetRecipeId(featureId),
      images: values.images,
    });
    const keyframeFields = enhanceKeyframeFields({
      keyframes: values.keyframes,
      duration: values.duration,
      fps: values.fps,
    });
    const videoAssetId = values.video?.assetId;
    const sourceVideoAssetId = videoAssetId ?? null;
    const replacedWindow =
      videoAssetId === undefined
        ? undefined
        : replacedWindowFor(featureId, {
            startTime: values.startTime,
            duration: values.duration,
          });
    if (isLoraRecipeId(featureId)) {
      loraCatalogIds.push(getLoraRecipe(featureId).catalogId);
    }
    if (featureId === CUSTOM_LORA_HOME_FEATURE.id && customLoraId) {
      customLoraIds.push(customLoraId);
    }
    const icLoraId = isIcLoraRecipeId(featureId)
      ? icLoraCatalogId(featureId)
      : undefined;

    try {
      const result = await api.enhancePrompt({
        prompt: sourcePrompt,
        negativePrompt: sourceNegativePrompt,
        loraCatalogIds,
        customLoraIds,
        icLoraId,
        imageAssetId,
        lastImageAssetId,
        imageAssetIds: panelAssetIds,
        ...(panelAssetIds !== undefined
          ? {
              aspectRatio: values.aspectRatio,
              resolution: values.resolution,
            }
          : {}),
        ...keyframeFields,
        videoAssetId,
        replacedWindow,
        provider: enhanceProvider,
        mediaType: "video",
      });

      if (!result.ok) {
        setEnhanceError(enhanceFailureMessage(result.error, onGeminiKeyRequired));
        return;
      }

      const live = liveFormState(featureId, storeKey);
      const history = live.history ?? promptHistory;
      const index = live.index ?? historyIndex;
      const liveNegative =
        nagUnavailableReason == null
          ? (live.negativePrompt ?? negativePromptRef.current)
          : "";
      if (
        !shouldApplyEnhanceResult(
          live.prompt ?? promptRef.current,
          sourcePrompt,
          liveNegative,
          sourceNegativePrompt,
        )
      ) {
        return;
      }
      if (
        !isEnhanceSourceCurrent(live, {
          videoAssetId: sourceVideoAssetId,
          startFrameAssetId,
          referenceImageAssetId: referenceImageAssetId ?? null,
          panelAssetIds: panelAssetIds ?? [],
          keyframes: keyframesSignature(values.keyframes, {
            duration: values.duration,
            fps: values.fps,
          }),
          replacedWindow,
        })
      ) {
        return;
      }

      const next = historyAfterEnhance({
        history,
        index,
        sourcePrompt,
        sourceNegativePrompt,
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
    customLoraId,
    enhanceProvider,
    featureId,
    hasEnhanceInput,
    historyIndex,
    isEnhancing,
    nagUnavailableReason,
    negativePrompt,
    onGeminiKeyRequired,
    onPromptHistoryChange,
    prompt,
    promptHistory,
    storeKey,
    values.aspectRatio,
    values.duration,
    values.fps,
    values.keyframes,
    values.endFrame?.assetId,
    values.image?.assetId,
    values.images,
    values.resolution,
    values.startFrame?.assetId,
    values.startTime,
    values.video?.assetId,
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
  });

  const promptProvenance: PromptProvenance = resolvePromptProvenance(
    prompt,
    promptHistory,
    historyIndex,
    negativePrompt,
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
    nagUnavailableReason,
    showNagStrengthControls,
  };
}
