import type { CustomLoraSchema } from "./definitions/customLora";
import { orderVideoSettingsOptionFields } from "./fields/optionControlPresentation";
import { getVideoOptionIcon } from "./fields/videoOptionIcons";
import type { FeatureFormPresentationAdapter } from "./FeatureFormView";

/** The same field order as every video form. */
export const customLoraFormPresentation: FeatureFormPresentationAdapter<CustomLoraSchema> = {
  getOptionIcon: getVideoOptionIcon,
  orderOptionFields: (fields) => orderVideoSettingsOptionFields(fields),
};
