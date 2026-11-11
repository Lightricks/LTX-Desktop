import type { ReactNode } from "react";

import formStyles from "../FeatureFormView.module.scss";
import type {
  FeatureFieldSchema,
  FieldOption,
  OptionsFieldDefForKey,
  OptionsFieldKey,
  ValidationIssue,
} from "../types";
import { FieldLabel } from "./FieldLabel";
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
      <FieldLabel label={field.label} tooltip={field.tooltip} />
      <div className={formStyles.fieldControl}>
        <FormOptionControl
          options={field.options}
          value={value}
          onChange={onChange}
          ariaLabel={field.label}
          forcePicker={field.forcePicker}
          maxOptionCount={field.maxOptionCount}
          getOptionIcon={getOptionIcon}
          showListIcons={field.dataKey === "aspectRatio"}
        />
      </div>
      {issue ? (
        <span className={formStyles.fieldError}>{issue.message}</span>
      ) : null}
    </div>
  );
}
