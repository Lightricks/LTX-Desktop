import type { TextToVideoSchema } from "./definitions/textToVideo.ts";
import { orderVideoSettingsOptionFields } from "./fields/optionControlPresentation.ts";
import type { OptionsFieldDef } from "./types.ts";

export function orderTextToVideoOptionFields(
  fields: readonly OptionsFieldDef<TextToVideoSchema>[],
): OptionsFieldDef<TextToVideoSchema>[] {
  return orderVideoSettingsOptionFields(fields);
}
