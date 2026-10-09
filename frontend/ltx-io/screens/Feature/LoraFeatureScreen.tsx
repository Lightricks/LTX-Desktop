import { useMemo } from "react";

import { useLoraCatalog } from "@/hooks/use-catalog";
import { getLoraRecipe, recipeHasEndFrame, type LoraRecipeId } from "@/lib/lora-recipes";

import {
  LORA_RECIPE_END_FRAME_SEEDS,
  LORA_RECIPE_START_FRAME_SEEDS,
  planExploreLoraRecipeSeed,
} from "../../hooks/exploreLoraRecipeSeed.ts";
import { useCreateLoraRecipe } from "../../hooks/useCreateLoraRecipe";
import { useExploreLoraRecipeSeed } from "../../hooks/useExploreLoraRecipeSeed";
import { useExploreRuntime } from "../../runtime/ExploreRuntime";
import { useGenerations } from "../../hooks/useGenerations";
import { useVideoGenerationModelSpecs } from "../../hooks/useVideoGenerationModelSpecs";
import { useFeatureFormStore } from "../../stores/featureFormStore";
import { CatalogInstallPreflight } from "./CatalogInstallPreflight";
import type { LoraRecipeContext } from "./definitions/loraRecipe";
import { createLoraRecipeDefinition } from "./definitions/loraRecipe";
import { FeatureFormScreen } from "./FeatureFormScreen";
import { loraRecipeFormPresentation } from "./LoraRecipeFormPresentation";
import { LORA_RECIPE_EXAMPLE_MEDIA } from "./loraRecipeExampleMedia";
import { LoraRecipeExampleEmpty } from "./results/LoraRecipeExampleEmpty";
import { ResultFrame } from "./results/ResultFrame";

/**
 * One screen for every listed LoRA recipe (no per-recipe screen). The shared
 * catalog preflight owns download, Hugging Face sign-in, and the phone gate.
 * i2v recipes always load the packaged start-frame still so Reset has a seed;
 * first-paint ingest still follows planExploreLoraRecipeSeed.
 * The create body is catalogId + scale only (plus startFrame for i2v).
 */
export function LoraFeatureScreen({ recipeId }: { recipeId: LoraRecipeId }) {
  const recipe = getLoraRecipe(recipeId);
  const { api } = useExploreRuntime();
  const { loras } = useLoraCatalog(true, api);
  const catalogItem = useMemo(
    () => loras.find((entry) => entry.lora.id === recipe.catalogId),
    [loras, recipe.catalogId],
  );
  const specsQuery = useVideoGenerationModelSpecs();
  const createGeneration = useCreateLoraRecipe(recipe.id);

  const hasStoredValues = useFeatureFormStore((state) =>
    state.hasValues(recipe.id),
  );
  const generationsQuery = useGenerations(recipe.id);
  const seedPlan = planExploreLoraRecipeSeed({
    mode: recipe.mode,
    hasStoredValues,
    generationsReady: generationsQuery.isSuccess,
    generationsFailed: generationsQuery.isError,
    lastGenerationSpec: generationsQuery.data?.[0]?.spec,
  });
  const needsEndFrame = recipeHasEndFrame(recipe);
  const seedId =
    recipe.mode === "i2v" ? LORA_RECIPE_START_FRAME_SEEDS[recipe.id] : undefined;
  const endSeedId = needsEndFrame
    ? LORA_RECIPE_END_FRAME_SEEDS[recipe.id]
    : undefined;
  const seedQuery = useExploreLoraRecipeSeed({
    seedId,
    enabled: recipe.mode === "i2v",
  });
  const endSeedQuery = useExploreLoraRecipeSeed({
    seedId: endSeedId,
    enabled: needsEndFrame,
  });

  const definition = useMemo(() => createLoraRecipeDefinition(recipe), [recipe]);
  const context = useMemo<LoraRecipeContext>(
    () => ({
      specs: specsQuery.data,
      seed: seedQuery.data ?? null,
      endSeed: endSeedQuery.data ?? null,
    }),
    [endSeedQuery.data, seedQuery.data, specsQuery.data],
  );
  const ExampleEmptyState = useMemo(() => {
    const media = LORA_RECIPE_EXAMPLE_MEDIA[recipe.id];
    return function RecipeExampleEmpty() {
      return (
        <LoraRecipeExampleEmpty
          videoUrl={media.videoUrl}
          posterUrl={media.posterUrl}
        />
      );
    };
  }, [recipe.id]);

  const contextReady =
    specsQuery.isSuccess &&
    (seedPlan === "skip" ||
      (seedPlan === "ingest" &&
        seedQuery.isSuccess &&
        (!needsEndFrame || endSeedQuery.isSuccess)));
  const resourceError =
    specsQuery.error ??
    (seedPlan === "skip" ? null : seedQuery.error) ??
    (seedPlan === "skip" || !needsEndFrame ? null : endSeedQuery.error) ??
    (hasStoredValues ? null : generationsQuery.error);

  return (
    <CatalogInstallPreflight
      kind="lora"
      catalogId={recipe.catalogId}
      title={recipe.title}
      hold={!contextReady && resourceError == null}
      renderForm={(controls) => (
        <FeatureFormScreen
          definition={definition}
          context={{
            ...context,
            catalogVariants: controls.variants,
            supportedModels: catalogItem?.lora.supported_models,
          }}
          contextReady={contextReady}
          resourceError={resourceError}
          createGeneration={createGeneration}
          ResultFrameAdapter={ResultFrame}
          EmptyStateAdapter={ExampleEmptyState}
          presentation={loraRecipeFormPresentation}
          generateLabel={controls.generateLabel}
          generateHint={controls.generateHint}
          generateBusy={controls.generateBusy}
          generateEnabled={controls.generateEnabled}
          actionError={controls.actionError}
          interceptGenerate={controls.interceptGenerate}
          onCancelGenerate={controls.onCancelGenerate}
        />
      )}
    />
  );
}
