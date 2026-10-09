import { create } from "zustand";

import type { HomeFeatureId } from "@/lib/home-features";
import type { IcLoraRecipeId } from "@/lib/ic-lora-recipes";
import type { LoraRecipeId } from "@/lib/lora-recipes";
import type { PromptHistoryState } from "@/lib/prompt-provenance";

import type { AudioToVideoSchema } from "../screens/Feature/definitions/audioToVideo";
import type { ImageToVideoSchema } from "../screens/Feature/definitions/imageToVideo";
import type { LoraRecipeSchema } from "../screens/Feature/definitions/loraRecipe";
import type { IcLoraRecipeSchema } from "../screens/Feature/definitions/icLoraRecipe";
import type { ExtendSchema } from "../screens/Feature/definitions/extend";
import type { RetakeSchema } from "../screens/Feature/definitions/retake";
import type { TextToVideoSchema } from "../screens/Feature/definitions/textToVideo";
import type { FeatureValues } from "../screens/Feature/types";

/** Closed map: adding a HomeFeatureId without a schema is a type error. */
export type FeatureSchemaMap = {
  [K in HomeFeatureId]: K extends "text-to-video"
    ? TextToVideoSchema
    : K extends "image-to-video"
      ? ImageToVideoSchema
      : K extends LoraRecipeId
        ? LoraRecipeSchema
        : K extends "audio-to-video"
          ? AudioToVideoSchema
          : K extends "retake"
            ? RetakeSchema
            : K extends "extend"
              ? ExtendSchema
              : K extends IcLoraRecipeId
                ? IcLoraRecipeSchema
                : never;
};

type FeatureFormStore = {
  valuesByFeature: Partial<{
    [K in HomeFeatureId]: FeatureValues<FeatureSchemaMap[K]>;
  }>;
  promptHistoryByFeature: Partial<Record<HomeFeatureId, PromptHistoryState>>;
  getValues: <K extends HomeFeatureId>(
    id: K,
  ) => FeatureValues<FeatureSchemaMap[K]> | undefined;
  hasValues: (id: HomeFeatureId) => boolean;
  setValues: <K extends HomeFeatureId>(
    id: K,
    values: FeatureValues<FeatureSchemaMap[K]>,
  ) => void;
  setPromptHistory: (id: HomeFeatureId, state: PromptHistoryState) => void;
};

export const useFeatureFormStore = create<FeatureFormStore>((set, get) => ({
  valuesByFeature: {},
  promptHistoryByFeature: {},
  getValues: (id) => get().valuesByFeature[id],
  hasValues: (id) => get().valuesByFeature[id] !== undefined,
  setValues: (id, values) =>
    set((state) => ({
      valuesByFeature: { ...state.valuesByFeature, [id]: values },
    })),
  setPromptHistory: (id, promptHistory) =>
    set((state) => ({
      promptHistoryByFeature: {
        ...state.promptHistoryByFeature,
        [id]: promptHistory,
      },
    })),
}));
