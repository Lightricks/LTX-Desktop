import { Text } from "@ds/Text/Text";

import formStyles from "../FeatureFormView.module.scss";
import type {
  FeatureFieldSchema,
  SliderFieldDef,
  ValidationIssue,
} from "../types";

export function SliderField<TSchema extends FeatureFieldSchema>({
  field,
  value,
  issue,
  onChange,
}: {
  field: SliderFieldDef<TSchema>;
  value: number;
  issue?: ValidationIssue;
  onChange: (value: number) => void;
}) {
  // No DS Slider today: a native range input, styled via accent-color.
  const decimals = field.step < 1 ? 2 : 0;
  return (
    <div className={formStyles.field}>
      <Text as="span" variant="body" size="md" className={formStyles.fieldLabel}>
        {field.label}
      </Text>
      <div className={formStyles.sliderControl}>
        <input
          type="range"
          className={formStyles.slider}
          min={field.min}
          max={field.max}
          step={field.step}
          value={value}
          aria-label={field.label}
          onChange={(event) => onChange(Number(event.currentTarget.value))}
        />
        <span className={formStyles.sliderValue}>{value.toFixed(decimals)}</span>
      </div>
      {issue ? (
        <span className={formStyles.fieldError}>{issue.message}</span>
      ) : null}
    </div>
  );
}
