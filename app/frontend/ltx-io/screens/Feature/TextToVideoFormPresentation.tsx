import type { TextToVideoSchema } from "./definitions/textToVideo";
import { getVideoOptionIcon } from "./fields/videoOptionIcons";
import type { FeatureFormPresentationAdapter } from "./FeatureFormView";
import { orderTextToVideoOptionFields } from "./textToVideoOptionFieldOrder";

export const textToVideoFormPresentation: FeatureFormPresentationAdapter<TextToVideoSchema> =
  {
    getOptionIcon: getVideoOptionIcon,
    orderOptionFields: orderTextToVideoOptionFields,
  };
