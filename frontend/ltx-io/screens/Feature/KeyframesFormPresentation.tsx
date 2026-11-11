import type { KeyframesSchema } from "./definitions/keyframes";
import type { FeatureFormPresentationAdapter } from "./FeatureFormView";
import { getVideoSettingsOptionIcon } from "./ImageToVideoFormPresentation";
import { orderKeyframesOptionFields } from "./keyframesOptionFieldOrder";

export const keyframesFormPresentation: FeatureFormPresentationAdapter<KeyframesSchema> =
  {
    getOptionIcon: getVideoSettingsOptionIcon,
    orderOptionFields: orderKeyframesOptionFields,
  };
