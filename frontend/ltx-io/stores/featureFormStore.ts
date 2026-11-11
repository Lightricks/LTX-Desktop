import { create } from "zustand";

import type { CustomLoraNavId } from "@/lib/custom-lora";
import type { HomeFeatureId } from "@/lib/home-features";
import type {
  IcLoraSheetRecipeId,
  IcLoraTrackRecipeId,
  IcLoraVideoRecipeId,
} from "@/lib/ic-lora-recipes";
import type { LoraRecipeId } from "@/lib/lora-recipes";
import type { PromptHistoryState } from "@/lib/prompt-provenance";

import type { NagStrength } from "../lib/nagStrength";

import type { CustomLoraSchema } from "../screens/Feature/definitions/customLora";
import type { AudioToVideoSchema } from "../screens/Feature/definitions/audioToVideo";
import type { ImageToVideoSchema } from "../screens/Feature/definitions/imageToVideo";
import type { KeyframesSchema } from "../screens/Feature/definitions/keyframes";
import type { LoraRecipeSchema } from "../screens/Feature/definitions/loraRecipe";
import type { IcLoraRecipeSchema } from "../screens/Feature/definitions/icLoraRecipe";
import type { IcLoraSheetSchema } from "../screens/Feature/definitions/icLoraSheet";
import type { IcLoraTrackSchema } from "../screens/Feature/definitions/icLoraTrack";
import type { ExtendSchema } from "../screens/Feature/definitions/extend";
import type { RetakeSchema } from "../screens/Feature/definitions/retake";
import type { TextToVideoSchema } from "../screens/Feature/definitions/textToVideo";
import type { FeatureValues } from "../screens/Feature/types";

/** Closed map: adding a HomeFeatureId without a schema is a type error. */
export type FeatureSchemaMap = {
  [K in HomeFeatureId]: K extends "text-to-video"
    ? TextToVideoSchema
    : K extends "custom-lora"
      ? CustomLoraSchema
      : K extends "image-to-video"
        ? ImageToVideoSchema
        : K extends "keyframes"
          ? KeyframesSchema
          : K extends LoraRecipeId
            ? LoraRecipeSchema
            : K extends "audio-to-video"
              ? AudioToVideoSchema
              : K extends "retake"
                ? RetakeSchema
                : K extends "extend"
                  ? ExtendSchema
                  : K extends IcLoraVideoRecipeId
                    ? IcLoraRecipeSchema
                    : K extends IcLoraSheetRecipeId
                      ? IcLoraSheetSchema
                      : K extends IcLoraTrackRecipeId
                        ? IcLoraTrackSchema
                        : never;
};

/** Each saved custom LoRA gets its own form slot (`custom-lora:<id>`). */
export type FeatureFormKey = HomeFeatureId | CustomLoraNavId;

export type FeatureSchemaForKey<K extends FeatureFormKey> = K extends HomeFeatureId
  ? FeatureSchemaMap[K]
  : CustomLoraSchema;

type FeatureFormStore = {
  valuesByFeature: Partial<{
    [K in FeatureFormKey]: FeatureValues<FeatureSchemaForKey<K>>;
  }>;
  promptHistoryByFeature: Partial<Record<FeatureFormKey, PromptHistoryState>>;
  getValues: <K extends FeatureFormKey>(
    id: K,
  ) => FeatureValues<FeatureSchemaForKey<K>> | undefined;
  hasValues: (id: FeatureFormKey) => boolean;
  setValues: <K extends FeatureFormKey>(
    id: K,
    values: FeatureValues<FeatureSchemaForKey<K>>,
  ) => void;
  setPromptHistory: (id: FeatureFormKey, state: PromptHistoryState) => void;
  /** Negative prompt strength per form, behind the "Per-run strength controls" setting. Unset means default. */
  nagStrengthByFeature: Partial<Record<FeatureFormKey, NagStrength>>;
  setNagStrength: (id: FeatureFormKey, strength: NagStrength) => void;
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
  nagStrengthByFeature: {},
  setNagStrength: (id, strength) =>
    set((state) => ({
      nagStrengthByFeature: { ...state.nagStrengthByFeature, [id]: strength },
    })),
}));
