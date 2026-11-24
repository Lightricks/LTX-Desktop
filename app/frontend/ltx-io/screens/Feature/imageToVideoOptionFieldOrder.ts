import type { ImageToVideoSchema } from "./definitions/imageToVideo.ts";
import { orderVideoSettingsOptionFields } from "./fields/optionControlPresentation.ts";
import type { OptionsFieldDef } from "./types.ts";

export function orderImageToVideoOptionFields(
  fields: readonly OptionsFieldDef<ImageToVideoSchema>[],
): OptionsFieldDef<ImageToVideoSchema>[] {
  return orderVideoSettingsOptionFields(fields);
}
