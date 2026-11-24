import { useMemo } from "react";

import { getIcLoraSheetRecipe, type IcLoraSheetRecipeId } from "@/lib/ic-lora-recipes";

import { CatalogInstallPreflight, type CatalogGenerateControls } from "./CatalogInstallPreflight";
import {
  createIcLoraSheetDefinition,
  readSheetImages,
  type IcLoraSheetContext,
  type IcLoraSheetSchema,
} from "./definitions/icLoraSheet";
import { getVideoOptionIcon } from "./fields/videoOptionIcons";
import { orderVideoSettingsOptionFields } from "./fields/optionControlPresentation";
import { FeatureFormScreen } from "./FeatureFormScreen";
import type { FeatureFormPresentationAdapter } from "./FeatureFormView";
import { ResultFrame } from "./results/ResultFrame";
import { useStillsIcLoraForm } from "./useStillsIcLoraForm";

/** The packaged pre-made sheet that fills an empty form. */
const SEED_IDS = ["ingredients-sheet"] as const;

const presentation: FeatureFormPresentationAdapter<IcLoraSheetSchema> = {
  getOptionIcon: getVideoOptionIcon,
  orderOptionFields: orderVideoSettingsOptionFields,
};

function IcLoraSheetForm({
  recipeId,
  controls,
}: {
  recipeId: IcLoraSheetRecipeId;
  controls: CatalogGenerateControls;
}) {
  const recipe = getIcLoraSheetRecipe(recipeId);
  const form = useStillsIcLoraForm({
    recipeId,
    catalogId: recipe.catalogId,
    seedIds: SEED_IDS,
    readSeeded: readSheetImages,
  });
  const definition = useMemo(
    () =>
      createIcLoraSheetDefinition(recipe, {
        audio_mode: controls.defaultAudioMode,
        default_negative_prompt: controls.defaultNegativePrompt,
      }),
    [recipe, controls.defaultAudioMode, controls.defaultNegativePrompt],
  );
  const context = useMemo<IcLoraSheetContext>(
    () => ({
      specs: form.specs,
      catalogVariants: controls.variants,
      supportedModels: form.supportedModels,
      seed: form.seed,
    }),
    [form.specs, controls.variants, form.supportedModels, form.seed],
  );
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

export function IcLoraSheetFeatureScreen({ recipeId }: { recipeId: IcLoraSheetRecipeId }) {
  const recipe = getIcLoraSheetRecipe(recipeId);
  return (
    <CatalogInstallPreflight
      kind="ic-lora"
      catalogId={recipe.catalogId}
      title={recipe.title}
      hold={false}
      renderForm={(controls) => <IcLoraSheetForm recipeId={recipeId} controls={controls} />}
    />
  );
}
