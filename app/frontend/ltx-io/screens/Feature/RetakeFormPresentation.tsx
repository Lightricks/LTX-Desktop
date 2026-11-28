import type { RetakeSchema } from "./definitions/retake";
import { getVideoOptionIcon } from "./fields/videoOptionIcons";
import type { FeatureFormPresentationAdapter } from "./FeatureFormView";
import { orderVideoSettingsOptionFields } from "./fields/optionControlPresentation";

export const retakeFormPresentation: FeatureFormPresentationAdapter<RetakeSchema> =
  {
    getOptionIcon: getVideoOptionIcon,
    orderOptionFields: orderVideoSettingsOptionFields,
  };
