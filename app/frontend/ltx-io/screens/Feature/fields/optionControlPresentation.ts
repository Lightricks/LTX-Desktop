import type { FieldOption } from "../types.ts";

export type OptionControlPresentation = "segment" | "picker" | "display-only";

export type OptionControlPresentationConfig = {
  forcePicker?: boolean;
  maxOptionCount?: number;
};

export const DISCRETE_OPTION_SEGMENT_THRESHOLD = 3;

export function shouldUseSegmentControl(
  options: readonly FieldOption[],
  config?: OptionControlPresentationConfig,
): boolean {
  if (config?.forcePicker) return false;
  if (options.length === 0) return false;
  const peakCount = Math.max(options.length, config?.maxOptionCount ?? 0);
  return peakCount <= DISCRETE_OPTION_SEGMENT_THRESHOLD;
}

export function getOptionControlPresentation(
  options: readonly FieldOption[],
  config?: OptionControlPresentationConfig,
): OptionControlPresentation {
  if (options.length <= 1) return "display-only";
  if (shouldUseSegmentControl(options, config)) return "segment";
  return "picker";
}

type OptionPresentationField = {
  options: readonly FieldOption[];
  forcePicker?: boolean;
  maxOptionCount?: number;
};

function getSettingsFieldSortRank(field: OptionPresentationField): number {
  switch (
    getOptionControlPresentation(field.options, {
      forcePicker: field.forcePicker,
      maxOptionCount: field.maxOptionCount,
    })
  ) {
    case "picker":
      return 0;
    case "display-only":
      return 1;
    case "segment":
      return 2;
  }
}

/** Pickers first, then read-only, then segments. Stable order within each group. */
export function sortSettingFields<TField extends OptionPresentationField>(
  fields: readonly TField[],
): TField[] {
  return fields
    .map((field, index) => ({ field, index }))
    .sort((a, b) => {
      const rankA = getSettingsFieldSortRank(a.field);
      const rankB = getSettingsFieldSortRank(b.field);
      if (rankA !== rankB) return rankA - rankB;
      return a.index - b.index;
    })
    .map(({ field }) => field);
}

export function orderVideoSettingsOptionFields<
  TField extends OptionPresentationField & { dataKey: string },
>(fields: readonly TField[]): TField[] {
  const modelFields = fields.filter((field) => field.dataKey === "model");
  const variantFields = fields.filter((field) => field.dataKey === "variant");
  const remainingFields = fields.filter(
    (field) => field.dataKey !== "model" && field.dataKey !== "variant",
  );
  return [...modelFields, ...variantFields, ...sortSettingFields(remainingFields)];
}

export function toOptionSegmentKey(value: FieldOption["value"]): string {
  return String(value);
}

export function isOptionSelected(
  value: unknown,
  option: FieldOption,
): boolean {
  return option.value === value;
}
