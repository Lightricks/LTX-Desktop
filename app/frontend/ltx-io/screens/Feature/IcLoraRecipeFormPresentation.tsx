import type { IcLoraRecipeSchema } from "./definitions/icLoraRecipe";
import { getVideoOptionIcon } from "./fields/videoOptionIcons";
import type { FeatureFormPresentationAdapter } from "./FeatureFormView";
import { orderVideoSettingsOptionFields } from "./fields/optionControlPresentation";

export const icLoraRecipeFormPresentation: FeatureFormPresentationAdapter<IcLoraRecipeSchema> =
  {
    getOptionIcon: getVideoOptionIcon,
    orderOptionFields: orderVideoSettingsOptionFields,
    naturalImagePreview: true,
  };
