import { useMemo } from "react";

import { getIcLoraTrackRecipe, type IcLoraTrackRecipeId } from "@/lib/ic-lora-recipes";
import { MOTION_TRACK_IMAGE_SEED } from "../../../../shared/explore-seed-filenames";

import { useImageAssetSize } from "../../hooks/useAsset";
import { useFeatureFormStore } from "../../stores/featureFormStore";
import { CatalogInstallPreflight, type CatalogGenerateControls } from "./CatalogInstallPreflight";
import {
  createIcLoraTrackDefinition,
  readTrack,
  type IcLoraTrackContext,
  type IcLoraTrackSchema,
} from "./definitions/icLoraTrack";
import { referenceImageAssetAction } from "./createWithAiImageActions";
import { getVideoOptionIcon } from "./fields/videoOptionIcons";
import { orderVideoSettingsOptionFields } from "./fields/optionControlPresentation";
import { FeatureFormScreen } from "./FeatureFormScreen";
import type { FeatureFormPresentationAdapter } from "./FeatureFormView";
import { ResultFrame } from "./results/ResultFrame";
import { useStillsIcLoraForm } from "./useStillsIcLoraForm";

/** The packaged still that fills an empty form. */
const SEED_IDS = [MOTION_TRACK_IMAGE_SEED.kind];

const presentation: FeatureFormPresentationAdapter<IcLoraTrackSchema> = {
  getOptionIcon: getVideoOptionIcon,
  orderOptionFields: orderVideoSettingsOptionFields,
  imageAssetAction: referenceImageAssetAction,
};

function IcLoraTrackForm({
  recipeId,
  controls,
}: {
  recipeId: IcLoraTrackRecipeId;
  controls: CatalogGenerateControls;
}) {
  const recipe = getIcLoraTrackRecipe(recipeId);
  const form = useStillsIcLoraForm({
    recipeId,
    catalogId: recipe.catalogId,
    seedIds: SEED_IDS,
    readSeeded: readTrack,
  });
  const startFrameId = useFeatureFormStore(
    (state) => state.getValues(recipeId)?.startFrame?.assetId ?? null,
  );
  const startFrameSize = useImageAssetSize(startFrameId);
  const definition = useMemo(
    () =>
      createIcLoraTrackDefinition(recipe, {
        audio_mode: controls.defaultAudioMode,
        default_negative_prompt: controls.defaultNegativePrompt,
      }),
    [recipe, controls.defaultAudioMode, controls.defaultNegativePrompt],
  );
  const context = useMemo<IcLoraTrackContext>(() => {
    return {
      specs: form.specs,
      catalogVariants: controls.variants,
      supportedModels: form.supportedModels,
      seed: form.seed?.[0],
      startFrameSize,
    };
  }, [form.specs, controls.variants, form.supportedModels, form.seed, startFrameSize]);
  if (!form.contextReady && form.resourceError == null) return null;
  return (
    <FeatureFormScreen
      definition={definition}
      context={context}
      contextReady={form.contextReady}
      resourceError={form.resourceError}
      createGeneration={form.createGeneration}
      ResultFrameAdapter={ResultFrame}
      EmptyStateAdapter={form.ExampleEmpty}
      presentation={presentation}
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

export function IcLoraTrackFeatureScreen({ recipeId }: { recipeId: IcLoraTrackRecipeId }) {
  const recipe = getIcLoraTrackRecipe(recipeId);
  return (
    <CatalogInstallPreflight
      kind="ic-lora"
      catalogId={recipe.catalogId}
      title={recipe.title}
      hold={false}
      renderForm={(controls) => <IcLoraTrackForm recipeId={recipeId} controls={controls} />}
    />
  );
}
