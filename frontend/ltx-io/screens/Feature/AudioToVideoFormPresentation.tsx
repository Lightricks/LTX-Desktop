import { getVideoOptionIcon } from "./fields/videoOptionIcons";
import { orderVideoSettingsOptionFields } from "./fields/optionControlPresentation";
import type { AudioToVideoSchema } from "./definitions/audioToVideo";
import type { FeatureFormPresentationAdapter } from "./FeatureFormView";

export const audioToVideoFormPresentation: FeatureFormPresentationAdapter<AudioToVideoSchema> =
  {
    getOptionIcon: getVideoOptionIcon,
    orderOptionFields: orderVideoSettingsOptionFields,
  };
