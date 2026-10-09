import { Text } from "@ds/Text/Text";
import type { ReactNode } from "react";

import formStyles from "../FeatureFormView.module.scss";
import type {
  FeatureFieldSchema,
  FieldOption,
  OptionsFieldDefForKey,
  OptionsFieldKey,
  ValidationIssue,
} from "../types";
import { FormOptionControl } from "./FormOptionControl";

export function SettingField<
  TSchema extends FeatureFieldSchema,
  TKey extends OptionsFieldKey<TSchema>,
>({
  field,
  value,
  issue,
  onChange,
  getOptionIcon,
}: {
  field: OptionsFieldDefForKey<TSchema, TKey>;
  value: Extract<TSchema[TKey]["value"], string | number>;
  issue?: ValidationIssue;
  onChange: (value: Extract<TSchema[TKey]["value"], string | number>) => void;
  getOptionIcon?: (
    option: FieldOption<Extract<TSchema[TKey]["value"], string | number>>,
  ) => ReactNode;
}) {
  return (
    <div className={formStyles.field}>
      <Text as="span" variant="body" size="md" className={formStyles.fieldLabel}>
        {field.label}
      </Text>
      <div className={formStyles.fieldControl}>
        <FormOptionControl
          options={field.options}
          value={value}
          onChange={onChange}
          ariaLabel={field.label}
          forcePicker={field.forcePicker}
          maxOptionCount={field.maxOptionCount}
          getOptionIcon={getOptionIcon}
        />
      </div>
      {issue ? (
        <span className={formStyles.fieldError}>{issue.message}</span>
      ) : null}
    </div>
  );
}
