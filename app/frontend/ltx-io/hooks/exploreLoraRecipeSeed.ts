import type { LoraEndFrameRecipeId, LoraI2vRecipeId } from "@/lib/lora-recipes";

import type { PackagedExploreSeedId } from "../assets/packaged-explore-assets";
import type { ExploreImageToVideoSeedPlan } from "./exploreImageToVideoSeed.ts";
import { readStartFrameFromGeneration } from "../screens/Feature/definitions/loraRecipeStartFrame.ts";

export type { ExploreImageToVideoSeedPlan };

/**
 * Packaged Start Frame still for every i2v recipe. First-paint ingest and Reset
 * both load this id. The Record is closed over `LoraI2vRecipeId` so a new i2v
 * recipe that forgets a seed is a type error, not a blank preflight.
 */
export const LORA_RECIPE_START_FRAME_SEEDS: Record<
  LoraI2vRecipeId,
  PackagedExploreSeedId
> = {
  cinemagraph: "cinemagraph-start-frame",
  "jib-up": "jib-up-start-frame",
  "jib-down": "jib-down-start-frame",
  "dolly-in": "dolly-in-start-frame",
  "dolly-out": "dolly-out-start-frame",
  transition: "transition-start-frame",
  vbvr: "vbvr-start-frame",
};

/**
 * Packaged End Frame still for every recipe that requires that slot.
 */
export const LORA_RECIPE_END_FRAME_SEEDS: Record<
  LoraEndFrameRecipeId,
  PackagedExploreSeedId
> = {
  transition: "transition-end-frame",
};

export function planExploreLoraRecipeSeed(input: {
  mode: "t2v" | "i2v" | "a2v";
  hasStoredValues: boolean;
  generationsReady: boolean;
  generationsFailed: boolean;
  lastGenerationSpec: unknown;
}): ExploreImageToVideoSeedPlan {
  if (input.mode !== "i2v") {
    return "skip";
  }
  if (input.hasStoredValues || input.generationsFailed) {
    return "skip";
  }
  if (!input.generationsReady) {
    return "wait";
  }
  if (readStartFrameFromGeneration(input.lastGenerationSpec) != null) {
    return "skip";
  }
  return "ingest";
}
