import { useMemo } from "react";

import { useIcLoras } from "@/hooks/use-catalog";
import {
  getIcLoraRecipe,
  type IcLoraRecipe,
  type IcLoraRecipeId,
} from "@/lib/ic-lora-recipes";

import { useExploreRuntime } from "../../runtime/ExploreRuntime";

import {
  isPackagedIcLoraSeedName,
  planIcLoraRecipeSeed,
  readAppliedIcLoraSeedRevision,
  writeAppliedIcLoraSeedRevision,
} from "../../hooks/icLoraRecipeSeed";
import { useIcLoraImageSeed } from "../../hooks/useIcLoraImageSeed";
import { usePackagedSeedAsset } from "../../hooks/usePackagedSeedAsset";
import { useCreateIcLoraRecipe } from "../../hooks/useCreateIcLoraRecipe";
import { CatalogInstallPreflight, type CatalogGenerateControls } from "./CatalogInstallPreflight";
import {
  createIcLoraRecipeDefinition,
  icLoraRecipeDefaults,
} from "./definitions/icLoraRecipe";
import { icLoraRecipeFormPresentation } from "./IcLoraRecipeFormPresentation";
import { IC_LORA_RECIPE_EXAMPLE_MEDIA } from "./icLoraRecipeExampleMedia";
import { FeatureFormScreen } from "./FeatureFormScreen";
import { ExampleEmptyMedia } from "./results/ExampleEmptyMedia";
import { ResultFrame, type ResultFrameProps } from "./results/ResultFrame";
import { useVideoToVideoFeature } from "./VideoToVideoFeatureScreen";

function CutoutResultFrame(props: ResultFrameProps) {
  return <ResultFrame {...props} isCutoutFeature />;
}

function IcLoraRecipeForm({
  recipe,
  controls,
}: {
  recipe: IcLoraRecipe;
  controls: CatalogGenerateControls;
}) {
  const { api } = useExploreRuntime();
  const { icLoras } = useIcLoras(true, api);
  const catalogItem = useMemo(
    () => icLoras.find((entry) => entry.ic_lora.id === recipe.catalogId),
    [icLoras, recipe.catalogId],
  );
  const allowsEmptyPrompt = catalogItem?.ic_lora.allows_empty_prompt ?? false;
  const settings = icLoraRecipeDefaults({
    lora_strength: controls.defaultLoraStrength,
    audio_mode: controls.defaultAudioMode,
  });
  const definition = useMemo(
    () =>
      createIcLoraRecipeDefinition(recipe, {
        lora_strength: settings.loraStrength,
        audio_mode: settings.audioMode,
        allows_empty_prompt: allowsEmptyPrompt,
      }),
    [recipe, settings.audioMode, settings.loraStrength, allowsEmptyPrompt],
  );
  const createGeneration = useCreateIcLoraRecipe(recipe.id);
  const seedKind = recipe.seed.kind;
  // The form waits for the look image seed, so the first stored values include it.
  const imageSeed = useIcLoraImageSeed(recipe);
  const useSeed = (options?: { enabled?: boolean }) =>
    usePackagedSeedAsset(seedKind, {
      enabled: (options?.enabled ?? true) && imageSeed.settled,
      revision: recipe.seed.revision,
      media: "video",
    });
  const ExampleEmpty = useMemo(() => {
    const media = IC_LORA_RECIPE_EXAMPLE_MEDIA[recipe.id];
    return function RecipeExampleEmpty() {
      return (
        <ExampleEmptyMedia posterUrl={media.posterUrl} videoUrl={media.videoUrl} />
      );
    };
  }, [recipe.id]);
  const screen = useVideoToVideoFeature({
    featureId: recipe.id,
    definition,
    presentation: icLoraRecipeFormPresentation,
    EmptyStateAdapter: ExampleEmpty,
    createGeneration,
    useSeed,
    planSeed: (input) =>
      planIcLoraRecipeSeed(recipe, {
        ...input,
        appliedSeedRevision: readAppliedIcLoraSeedRevision(recipe),
        packagedSeedRevision: recipe.seed.revision,
      }),
    isPackagedSeedName: (name) => isPackagedIcLoraSeedName(recipe, name),
    writeAppliedRevision: () => writeAppliedIcLoraSeedRevision(recipe),
    seedValues: (context) =>
      definition.initialValues?.({ ...context, imageSeed: imageSeed.data ?? null }) ??
      definition.defaults,
  });
  // A failed look image must show an error. The form needs it to run.
  const resourceError = screen.resourceError ?? imageSeed.error ?? null;
  const contextReady = screen.contextReady && imageSeed.settled;
  if (!contextReady && resourceError == null) return null;
  return (
    <FeatureFormScreen
      {...screen}
      contextReady={contextReady}
      resourceError={resourceError}
      context={{
        ...screen.context,
        catalogVariants: controls.variants,
        supportedModels: catalogItem?.ic_lora.supported_models,
        imageSeed: imageSeed.data ?? null,
        referenceImageRequired: catalogItem?.ic_lora.reference_image_required,
      }}
      ResultFrameAdapter={recipe.cutout ? CutoutResultFrame : ResultFrame}
      generateLabel={controls.generateLabel}
      generateHint={controls.generateHint}
      generateBusy={controls.generateBusy}
      generateEnabled={controls.generateEnabled}
      actionError={controls.actionError}
      interceptGenerate={controls.interceptGenerate}
      onCancelGenerate={controls.onCancelGenerate}
    />
  );
}

export function IcLoraRecipeFeatureScreen({ recipeId }: { recipeId: IcLoraRecipeId }) {
  const recipe = getIcLoraRecipe(recipeId);
  return (
    <CatalogInstallPreflight
      kind="ic-lora"
      catalogId={recipe.catalogId}
      title={recipe.title}
      hold={false}
      renderForm={(controls) => <IcLoraRecipeForm recipe={recipe} controls={controls} />}
    />
  );
}
