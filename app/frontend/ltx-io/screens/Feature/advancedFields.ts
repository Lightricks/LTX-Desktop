import type { FeatureFieldSchema, FeatureFormModel, FeatureValues } from "./types.ts";

/**
 * The form with its Advanced section removed while the user setting is off. A definition
 * lists its advanced fields and never reads the setting.
 */
export function formWithAdvancedFields<TSchema extends FeatureFieldSchema>(
  form: FeatureFormModel<TSchema>,
  enabled: boolean,
): FeatureFormModel<TSchema> {
  if (enabled || form.advancedFields == null) return form;
  return { ...form, advancedFields: [] };
}

/**
 * The values a run uses. While the setting is off, every advanced field reads its default,
 * so a stored advanced choice is neither shown nor sent. Returns the same object when the
 * setting is on or the form has no advanced field.
 *
 * TODO(security): CWE-602. This setting only changes the create body the form sends.
 * The backend still accepts a tiling override from a direct call. The range checks on
 * that request (HTTP 422) are the server control.
 */
export function valuesForRun<TSchema extends FeatureFieldSchema>(
  form: FeatureFormModel<TSchema>,
  values: FeatureValues<TSchema>,
  defaults: FeatureValues<TSchema>,
  enabled: boolean,
): FeatureValues<TSchema> {
  if (enabled || form.advancedFields == null || form.advancedFields.length === 0) return values;
  const next = { ...values };
  for (const field of form.advancedFields) {
    next[field.dataKey] = defaults[field.dataKey];
  }
  return next;
}
