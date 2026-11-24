import type { ExtendSchema } from "./definitions/extend";
import { getVideoOptionIcon } from "./fields/videoOptionIcons";
import type { FeatureFormPresentationAdapter } from "./FeatureFormView";
import { orderVideoSettingsOptionFields } from "./fields/optionControlPresentation";

export const extendFormPresentation: FeatureFormPresentationAdapter<ExtendSchema> =
  {
    getOptionIcon: getVideoOptionIcon,
    orderOptionFields: orderVideoSettingsOptionFields,
  };
