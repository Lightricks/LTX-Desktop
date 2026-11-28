import type { IcLoraRecipe } from "@/lib/ic-lora-recipes";

import { icLoraRecipeDefinition } from "../screens/Feature/definitions/icLoraRecipe.ts";
import { readGenerationInputs } from "../screens/Feature/definitions/videoFeature.ts";
import {
  isPackagedSeedName,
  planPackagedSeed,
  readAppliedSeedRevision,
  writeAppliedSeedRevision,
  type PackagedSeedPlan,
} from "./packagedSeed.ts";
function seedNames(recipe: IcLoraRecipe): readonly string[] {
  return [recipe.seed.filename, ...recipe.seed.previousFilenames];
}

export function planIcLoraRecipeSeed(
  recipe: IcLoraRecipe,
  input: {
    hasStoredValues: boolean;
    generationsReady: boolean;
    generationsFailed: boolean;
    lastGenerationSpec: unknown;
    hydratedVideoName?: string | null;
    hydratedVideoPending?: boolean;
    appliedSeedRevision?: number;
    packagedSeedRevision?: number;
  },
): PackagedSeedPlan {
  const definition = icLoraRecipeDefinition(recipe.id);
  return planPackagedSeed({
    hasStoredValues: input.hasStoredValues,
    generationsReady: input.generationsReady,
    generationsFailed: input.generationsFailed,
    lastGenerationSpec: input.hasStoredValues ? undefined : input.lastGenerationSpec,
    fromGeneration: (spec) => definition.fromGeneration(spec, {
      specs: null,
      seed: null,
      videoDurationSeconds: null,
      videoDurationPending: false,
      videoWidth: null,
      videoHeight: null,
      videoFps: null,
    }),
    hydratedName: input.hydratedVideoName ?? null,
    hydratedPending: input.hydratedVideoPending ?? false,
    appliedSeedRevision: input.appliedSeedRevision ?? recipe.seed.revision,
    packagedSeedRevision: input.packagedSeedRevision ?? recipe.seed.revision,
    packagedFilenames: seedNames(recipe),
  });
}

/**
 * Plans the packaged look image with the same planner as the video seed.
 * The result is "skip" for a recipe with no packaged image.
 */
export function planIcLoraImageSeed(
  recipe: IcLoraRecipe,
  input: {
    hasStoredValues: boolean;
    storedImageId: string | null;
    generationsReady: boolean;
    generationsFailed: boolean;
    lastGenerationSpec: unknown;
    hydratedImageName: string | null;
    hydratedImagePending: boolean;
    appliedSeedRevision: number;
  },
): PackagedSeedPlan {
  const seed = recipe.referenceImage?.seed;
  if (seed == null) return "skip";
  // Stored values with no image count as a choice only after the packaged image was
  // applied once. Before that, the image is still missing.
  const imageIsSettled =
    input.hasStoredValues &&
    (input.storedImageId != null || input.appliedSeedRevision >= seed.revision);
  return planPackagedSeed({
    hasStoredValues: imageIsSettled,
    generationsReady: input.generationsReady,
    generationsFailed: input.generationsFailed,
    lastGenerationSpec: input.hasStoredValues ? undefined : input.lastGenerationSpec,
    // A saved empty image is a choice too, so a present key counts.
    fromGeneration: (spec) => readGenerationInputs(spec, "image"),
    hydratedName: input.hydratedImageName,
    hydratedPending: input.hydratedImagePending,
    appliedSeedRevision: input.appliedSeedRevision,
    packagedSeedRevision: seed.revision,
    packagedFilenames: [seed.filename, ...seed.previousFilenames],
  });
}

export function isPackagedIcLoraSeedName(
  recipe: IcLoraRecipe,
  name: string | null | undefined,
): boolean {
  return isPackagedSeedName(name, seedNames(recipe));
}

export function readAppliedIcLoraSeedRevision(recipe: IcLoraRecipe): number {
  return readAppliedSeedRevision(recipe.seed.storageKey);
}

export function writeAppliedIcLoraSeedRevision(recipe: IcLoraRecipe): void {
  writeAppliedSeedRevision(recipe.seed.storageKey, recipe.seed.revision);
}

function imageRevisionKey(recipe: IcLoraRecipe): string {
  return `${recipe.seed.storageKey}:image`;
}

export function readAppliedIcLoraImageRevision(recipe: IcLoraRecipe): number {
  return readAppliedSeedRevision(imageRevisionKey(recipe));
}

export function writeAppliedIcLoraImageRevision(recipe: IcLoraRecipe): void {
  const seed = recipe.referenceImage?.seed;
  if (seed != null) writeAppliedSeedRevision(imageRevisionKey(recipe), seed.revision);
}
