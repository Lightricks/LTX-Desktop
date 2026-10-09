import type { LoraRecipeSchema } from "./definitions/loraRecipe";
import { orderVideoSettingsOptionFields } from "./fields/optionControlPresentation";
import { getVideoOptionIcon } from "./fields/videoOptionIcons";
import type { FeatureFormPresentationAdapter } from "./FeatureFormView";

export const loraRecipeFormPresentation: FeatureFormPresentationAdapter<LoraRecipeSchema> =
  {
    getOptionIcon: getVideoOptionIcon,
    orderOptionFields: (fields) => orderVideoSettingsOptionFields(fields),
  };
