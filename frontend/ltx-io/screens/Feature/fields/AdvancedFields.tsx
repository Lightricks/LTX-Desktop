import { AccordionArrow, AccordionMenu } from "@ds/AccordionMenu/AccordionMenu";
import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";

import { optionControlValue } from "../featureFormDisplay";
import {
  createFeatureFieldChange,
  readOptionsFieldValue,
  type FeatureFieldChange,
  type AdvancedFormField,
  type FeatureFieldSchema,
  type FeatureValues,
  type OptionsFieldKey,
  type ValidationIssue,
} from "../types";
import styles from "./AdvancedFields.module.scss";
import { SettingField } from "./SettingField";

type AdvancedFieldsProps<TSchema extends FeatureFieldSchema> = {
  fields: readonly AdvancedFormField<TSchema>[];
  values: FeatureValues<TSchema>;
  revealedIssues: readonly ValidationIssue[];
  onFieldChange: (...changes: FeatureFieldChange<TSchema>[]) => void;
};

/**
 * The collapsed Advanced section. It shows the options fields of `form.advancedFields`
 * with the same controls as the main form. The first option of each is Auto, and the
 * `defaultValue` of each is that option. A field with a tooltip shows it next to the label. A
 * form without advanced fields shows nothing. `useFeatureForm` also hides the fields while the
 * user setting is off.
 */
export function AdvancedFields<TSchema extends FeatureFieldSchema>({
  fields,
  values,
  revealedIssues,
  onFieldChange,
}: AdvancedFieldsProps<TSchema>) {
  if (fields.length === 0) return null;
  const issueOf = (fieldId: string) =>
    revealedIssues.find((candidate) => candidate.fieldId === fieldId);
  const shownValue = (field: AdvancedFormField<TSchema>) =>
    optionControlValue(
      readOptionsFieldValue(values, field.dataKey),
      field.options,
      field.defaultValue,
    );
  const changeFor = (field: AdvancedFormField<TSchema>, value: string | number) =>
    createFeatureFieldChange<TSchema, OptionsFieldKey<TSchema>>({
      kind: "options",
      fieldId: field.id,
      dataKey: field.dataKey,
      value,
    });
  const notAuto = fields.filter((field) => shownValue(field) !== field.defaultValue);
  return (
    <AccordionMenu
      trigger={
        <button type="button" className={styles.trigger}>
          <Text as="span" variant="body" size="md">
            Advanced
          </Text>
          <AccordionArrow />
        </button>
      }
    >
      <div className={styles.fields}>
        {fields.map((field) => (
          <SettingField
            key={field.id}
            field={field}
            value={shownValue(field)}
            issue={issueOf(field.id)}
            onChange={(value) => onFieldChange(changeFor(field, value))}
          />
        ))}
        {notAuto.length > 0 && (
          <div className={styles.reset}>
            <Button
              appearance="neutral"
              hierarchy="plain"
              size="sm"
              label="Reset to Auto"
              onClick={() => {
                // One call: the form folds every change onto the same values, so a call per
                // field would keep only the last one.
                onFieldChange(...notAuto.map((field) => changeFor(field, field.defaultValue)));
              }}
            />
          </div>
        )}
      </div>
    </AccordionMenu>
  );
}
