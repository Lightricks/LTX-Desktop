import type { KeyframesSchema } from "./definitions/keyframes.ts";
import { orderVideoSettingsOptionFields } from "./fields/optionControlPresentation.ts";
import type { OptionsFieldDef } from "./types.ts";

export function orderKeyframesOptionFields(
  fields: readonly OptionsFieldDef<KeyframesSchema>[],
): OptionsFieldDef<KeyframesSchema>[] {
  return orderVideoSettingsOptionFields(fields);
}
