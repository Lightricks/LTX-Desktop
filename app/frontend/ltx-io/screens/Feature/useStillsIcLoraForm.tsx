import { useMemo } from "react";

import { useIcLoras } from "@/hooks/use-catalog";
import type { IcLoraRecipeId } from "@/lib/ic-lora-recipes";

import type { PackagedExploreSeedId } from "../../assets/packaged-explore-assets";
import { planPackagedSeed } from "../../hooks/packagedSeed";
import { useCreateIcLoraRecipe } from "../../hooks/useCreateIcLoraRecipe";
import { useGenerations } from "../../hooks/useGenerations";
import { usePackagedSeedAssets } from "../../hooks/usePackagedSeedAsset";
import { useVideoGenerationModelSpecs } from "../../hooks/useVideoGenerationModelSpecs";
import { useExploreRuntime } from "../../runtime/ExploreRuntime";
import { useFeatureFormStore } from "../../stores/featureFormStore";
import { IC_LORA_RECIPE_EXAMPLE_MEDIA } from "./icLoraRecipeExampleMedia";
import { ExampleEmptyMedia } from "./results/ExampleEmptyMedia";

/**
 * What an IC-LoRA form needs when its input is packaged stills (a sheet, a track): the model
 * specs, the catalog entry, the stills that fill an empty form, and the example media.
 */
export function useStillsIcLoraForm({
  recipeId,
  catalogId,
  seedIds,
  readSeeded,
}: {
  recipeId: IcLoraRecipeId;
  catalogId: string;
  /** The packaged stills that fill an empty form, in order. */
  seedIds: readonly PackagedExploreSeedId[];
  /** Non-null when a saved generation spec holds the inputs the stills would fill. */
  readSeeded: (spec: unknown) => unknown;
}) {
  const { api } = useExploreRuntime();
  const { icLoras } = useIcLoras(true, api);
  const specsQuery = useVideoGenerationModelSpecs();
  const supportedModels = icLoras.find((entry) => entry.ic_lora.id === catalogId)?.ic_lora
    .supported_models;
  const createGeneration = useCreateIcLoraRecipe(recipeId);
  const hasStoredValues = useFeatureFormStore((state) => state.hasValues(recipeId));
  const generationsQuery = useGenerations(recipeId);
  const seedPlan = planPackagedSeed({
    hasStoredValues,
    generationsReady: generationsQuery.isSuccess,
    generationsFailed: generationsQuery.isError,
    lastGenerationSpec: generationsQuery.data?.[0]?.spec,
    fromGeneration: readSeeded,
  });
  const seedQuery = usePackagedSeedAssets(seedIds, {
    // Reset needs the stills too, so they load unless the plan still waits.
    enabled: seedPlan !== "wait",
    media: "image",
  });
  const contextReady =
    specsQuery.isSuccess &&
    (seedPlan === "skip" || (seedPlan === "ingest" && seedQuery.isSuccess));
  const resourceError =
    specsQuery.error ??
    (seedPlan === "ingest" ? seedQuery.error : null) ??
    (hasStoredValues ? null : generationsQuery.error);
  const ExampleEmpty = useMemo(() => {
    const media = IC_LORA_RECIPE_EXAMPLE_MEDIA[recipeId];
    return function StillsExampleEmpty() {
      return <ExampleEmptyMedia posterUrl={media.posterUrl} videoUrl={media.videoUrl} />;
    };
  }, [recipeId]);
  return {
    specs: specsQuery.data,
    supportedModels,
    seed: seedQuery.data,
    createGeneration,
    contextReady,
    resourceError,
    ExampleEmpty,
  };
}
