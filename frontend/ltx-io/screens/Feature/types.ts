import type { HomeFeatureId } from "../../../lib/home-features";

export type FieldOption<TValue extends string | number = string | number> = {
  value: TValue;
  label: string;
  /** Soft caveat shown on hover, e.g. untrained aspect ratios. */
  warning?: string;
  /** The option is listed but cannot be picked. The text says why. */
  disabledReason?: string;
};

export type AssetRef = {
  assetId: string;
};

export function toDurableAssetRef(asset: { id: string }): AssetRef {
  return { assetId: asset.id };
}

export type FeatureFieldKind =
  | "textarea"
  | "options"
  | "image-asset"
  | "slider"
  | "audio-asset"
  | "video-asset";

export type FeatureFieldSpec =
  | { kind: "textarea"; value?: never }
  | { kind: "options"; value: string | number | null }
  | { kind: "image-asset"; value: AssetRef | null }
  | { kind: "slider"; value: number }
  | { kind: "audio-asset"; value: AssetRef | null }
  | { kind: "video-asset"; value: AssetRef | null };

export type FeatureFieldSchema = Record<string, FeatureFieldSpec>;

export type FeatureFieldValue<TSpec extends FeatureFieldSpec> = TSpec extends {
  kind: "textarea";
}
  ? string
  : TSpec["value"];

export type FeatureValues<TSchema extends FeatureFieldSchema> =
  TSchema extends unknown
    ? { [TKey in keyof TSchema]: FeatureFieldValue<TSchema[TKey]> }
    : never;

export type FieldKeyOfKind<
  TSchema extends FeatureFieldSchema,
  TKind extends FeatureFieldKind,
> = Extract<
  {
    [TKey in keyof TSchema]: TSchema[TKey]["kind"] extends TKind ? TKey : never;
  }[keyof TSchema],
  string
>;

export type TextareaFieldKey<TSchema extends FeatureFieldSchema> =
  FieldKeyOfKind<TSchema, "textarea">;

export type OptionsFieldKey<TSchema extends FeatureFieldSchema> =
  FieldKeyOfKind<TSchema, "options">;

export type ImageAssetFieldKey<TSchema extends FeatureFieldSchema> =
  FieldKeyOfKind<TSchema, "image-asset">;

export type SliderFieldKey<TSchema extends FeatureFieldSchema> =
  FieldKeyOfKind<TSchema, "slider">;

export type AudioAssetFieldKey<TSchema extends FeatureFieldSchema> =
  FieldKeyOfKind<TSchema, "audio-asset">;

export type VideoAssetFieldKey<TSchema extends FeatureFieldSchema> =
  FieldKeyOfKind<TSchema, "video-asset">;

type FeatureFieldBase<TKey extends string> = {
  id: string;
  label: string;
  dataKey: TKey;
};

export type TextareaFieldDefForKey<
  TSchema extends FeatureFieldSchema,
  TKey extends TextareaFieldKey<TSchema>,
> = FeatureFieldBase<TKey> & {
  kind: "textarea";
  placeholder?: string;
};

export type TextareaFieldDef<TSchema extends FeatureFieldSchema> = {
  [TKey in TextareaFieldKey<TSchema>]: TextareaFieldDefForKey<TSchema, TKey>;
}[TextareaFieldKey<TSchema>];

export type OptionsFieldDefForKey<
  TSchema extends FeatureFieldSchema,
  TKey extends OptionsFieldKey<TSchema>,
> = FeatureFieldBase<TKey> & {
  kind: "options";
  options: FieldOption<Extract<TSchema[TKey]["value"], string | number>>[];
  /**
   * The option the control shows while the stored value is null. Without it the control
   * shows the first option. A field that stores "Original" as null sets it to Original.
   */
  defaultValue?: Extract<TSchema[TKey]["value"], string | number>;
  forcePicker?: boolean;
  maxOptionCount?: number;
};

export type OptionsFieldDef<TSchema extends FeatureFieldSchema> = {
  [TKey in OptionsFieldKey<TSchema>]: OptionsFieldDefForKey<TSchema, TKey>;
}[OptionsFieldKey<TSchema>];

export type ImageAssetFieldDefForKey<
  TSchema extends FeatureFieldSchema,
  TKey extends ImageAssetFieldKey<TSchema>,
> = FeatureFieldBase<TKey> & {
  kind: "image-asset";
};

export type ImageAssetFieldDef<TSchema extends FeatureFieldSchema> = {
  [TKey in ImageAssetFieldKey<TSchema>]: ImageAssetFieldDefForKey<
    TSchema,
    TKey
  >;
}[ImageAssetFieldKey<TSchema>];

export type SliderFieldDefForKey<
  TSchema extends FeatureFieldSchema,
  TKey extends SliderFieldKey<TSchema>,
> = FeatureFieldBase<TKey> & {
  kind: "slider";
  min: number;
  max: number;
  step: number;
};

export type SliderFieldDef<TSchema extends FeatureFieldSchema> = {
  [TKey in SliderFieldKey<TSchema>]: SliderFieldDefForKey<TSchema, TKey>;
}[SliderFieldKey<TSchema>];

export type AudioAssetFieldDefForKey<
  TSchema extends FeatureFieldSchema,
  TKey extends AudioAssetFieldKey<TSchema>,
> = FeatureFieldBase<TKey> & {
  kind: "audio-asset";
};

export type AudioAssetFieldDef<TSchema extends FeatureFieldSchema> = {
  [TKey in AudioAssetFieldKey<TSchema>]: AudioAssetFieldDefForKey<
    TSchema,
    TKey
  >;
}[AudioAssetFieldKey<TSchema>];

export type VideoAssetFieldDefForKey<
  TSchema extends FeatureFieldSchema,
  TKey extends VideoAssetFieldKey<TSchema>,
> = FeatureFieldBase<TKey> & {
  kind: "video-asset";
  /** Longer uploads open the trim modal before they are accepted. */
  maxDurationSeconds?: number;
};

export type VideoAssetFieldDef<TSchema extends FeatureFieldSchema> = {
  [TKey in VideoAssetFieldKey<TSchema>]: VideoAssetFieldDefForKey<
    TSchema,
    TKey
  >;
}[VideoAssetFieldKey<TSchema>];

export type FeatureFormField<TSchema extends FeatureFieldSchema> =
  | TextareaFieldDef<TSchema>
  | OptionsFieldDef<TSchema>
  | ImageAssetFieldDef<TSchema>
  | SliderFieldDef<TSchema>
  | AudioAssetFieldDef<TSchema>
  | VideoAssetFieldDef<TSchema>;

export type FeatureFormModel<TSchema extends FeatureFieldSchema> = {
  fields: FeatureFormField<TSchema>[];
};

export type FeatureFieldChange<TSchema extends FeatureFieldSchema> = {
  [TKey in keyof TSchema & string]: {
    kind: TSchema[TKey]["kind"];
    fieldId: string;
    dataKey: TKey;
    value: FeatureFieldValue<TSchema[TKey]>;
  };
}[keyof TSchema & string];

export function createFeatureFieldChange<
  TSchema extends FeatureFieldSchema,
  TKey extends TextareaFieldKey<TSchema>,
>(change: {
  kind: "textarea";
  fieldId: string;
  dataKey: TKey;
  value: string;
}): FeatureFieldChange<TSchema>;
export function createFeatureFieldChange<
  TSchema extends FeatureFieldSchema,
  TKey extends OptionsFieldKey<TSchema>,
>(change: {
  kind: "options";
  fieldId: string;
  dataKey: TKey;
  value: TSchema[TKey]["value"];
}): FeatureFieldChange<TSchema>;
export function createFeatureFieldChange<
  TSchema extends FeatureFieldSchema,
  TKey extends ImageAssetFieldKey<TSchema>,
>(change: {
  kind: "image-asset";
  fieldId: string;
  dataKey: TKey;
  value: TSchema[TKey]["value"];
}): FeatureFieldChange<TSchema>;
export function createFeatureFieldChange<
  TSchema extends FeatureFieldSchema,
  TKey extends SliderFieldKey<TSchema>,
>(change: {
  kind: "slider";
  fieldId: string;
  dataKey: TKey;
  value: number;
}): FeatureFieldChange<TSchema>;
export function createFeatureFieldChange<
  TSchema extends FeatureFieldSchema,
  TKey extends AudioAssetFieldKey<TSchema>,
>(change: {
  kind: "audio-asset";
  fieldId: string;
  dataKey: TKey;
  value: TSchema[TKey]["value"];
}): FeatureFieldChange<TSchema>;
export function createFeatureFieldChange<
  TSchema extends FeatureFieldSchema,
  TKey extends VideoAssetFieldKey<TSchema>,
>(change: {
  kind: "video-asset";
  fieldId: string;
  dataKey: TKey;
  value: TSchema[TKey]["value"];
}): FeatureFieldChange<TSchema>;
export function createFeatureFieldChange<
  TSchema extends FeatureFieldSchema,
  TKey extends keyof TSchema & string,
>(change: {
  kind: TSchema[TKey]["kind"];
  fieldId: string;
  dataKey: TKey;
  value: FeatureFieldValue<TSchema[TKey]>;
}): FeatureFieldChange<TSchema> {
  // Overloads already correlate kind, dataKey, and value from TSchema;
  // the implementation signature cannot re-prove that mapped union.
  return change as FeatureFieldChange<TSchema>;
}

export function readConstrainedFieldValue<
  TSchema extends FeatureFieldSchema,
  TKey extends keyof FeatureValues<TSchema> & string,
  TValue,
>(
  values: FeatureValues<TSchema>,
  dataKey: TKey,
): Extract<FeatureValues<TSchema>[TKey], TValue> {
  // Generic FeatureValues[TKey] stays the full schema value union even when
  // TKey is constrained to a narrower field kind.
  return values[dataKey] as Extract<FeatureValues<TSchema>[TKey], TValue>;
}

export function readOptionsFieldValue<
  TSchema extends FeatureFieldSchema,
  TKey extends OptionsFieldKey<TSchema>,
>(
  values: FeatureValues<TSchema>,
  dataKey: TKey,
): Extract<TSchema[TKey]["value"], string | number> {
  return readConstrainedFieldValue<TSchema, TKey, string | number>(
    values,
    dataKey,
  );
}

export function readImageAssetFieldValue<
  TSchema extends FeatureFieldSchema,
  TKey extends ImageAssetFieldKey<TSchema>,
>(values: FeatureValues<TSchema>, dataKey: TKey): AssetRef | null {
  return readConstrainedFieldValue<TSchema, TKey, AssetRef | null>(
    values,
    dataKey,
  );
}

export function readSliderFieldValue<
  TSchema extends FeatureFieldSchema,
  TKey extends SliderFieldKey<TSchema>,
>(values: FeatureValues<TSchema>, dataKey: TKey): number {
  return readConstrainedFieldValue<TSchema, TKey, number>(values, dataKey);
}

export function readAudioAssetFieldValue<
  TSchema extends FeatureFieldSchema,
  TKey extends AudioAssetFieldKey<TSchema>,
>(values: FeatureValues<TSchema>, dataKey: TKey): AssetRef | null {
  return readConstrainedFieldValue<TSchema, TKey, AssetRef | null>(
    values,
    dataKey,
  );
}

export function readVideoAssetFieldValue<
  TSchema extends FeatureFieldSchema,
  TKey extends VideoAssetFieldKey<TSchema>,
>(values: FeatureValues<TSchema>, dataKey: TKey): AssetRef | null {
  return readConstrainedFieldValue<TSchema, TKey, AssetRef | null>(
    values,
    dataKey,
  );
}

export function applyFeatureFieldChange<TSchema extends FeatureFieldSchema>(
  values: FeatureValues<TSchema>,
  change: FeatureFieldChange<TSchema>,
): FeatureValues<TSchema> {
  return { ...values, [change.dataKey]: change.value };
}

export type ValidationIssue = {
  id: string;
  fieldId: string;
  message: string;
  /**
   * Show without waiting for a Generate press. Use for issues that describe a
   * value the user already supplied. Blocking issues also disable Generate, so
   * the press that would otherwise reveal them can never happen.
   */
  alwaysRevealed?: boolean;
  /**
   * When false, Generate can proceed while the message stays visible.
   * Defaults to blocking.
   */
  blocksGenerate?: boolean;
};

export type FeatureDefinition<
  TSchema extends FeatureFieldSchema,
  TCreateBody,
  TContext,
  TId extends HomeFeatureId = HomeFeatureId,
> = {
  id: TId;
  title: string;
  defaults: FeatureValues<TSchema>;
  initialValues?: (context: TContext) => FeatureValues<TSchema>;
  resetValues?: (context: TContext) => FeatureValues<TSchema>;
  form: (
    values: FeatureValues<TSchema>,
    context: TContext,
  ) => FeatureFormModel<TSchema>;
  applyChange: (
    values: FeatureValues<TSchema>,
    change: FeatureFieldChange<TSchema>,
    context: TContext,
  ) => FeatureValues<TSchema>;
  normalize: (
    values: FeatureValues<TSchema>,
    context: TContext,
  ) => FeatureValues<TSchema>;
  validate: (
    values: FeatureValues<TSchema>,
    context: TContext,
  ) => ValidationIssue[];
  fromGeneration: (
    spec: unknown,
    context: TContext,
  ) => FeatureValues<TSchema> | null;
  toCreateBody: (
    values: FeatureValues<TSchema>,
    context: TContext,
  ) => TCreateBody;
  isReady?: (values: FeatureValues<TSchema>, context: TContext) => boolean;
  isUnavailable?: (
    values: FeatureValues<TSchema>,
    context: TContext,
  ) => boolean;
};

export function resolveFeatureSeedValues<
  TSchema extends FeatureFieldSchema,
  TCreateBody,
  TContext,
>(
  definition: Pick<
    FeatureDefinition<TSchema, TCreateBody, TContext>,
    "defaults" | "initialValues" | "normalize" | "fromGeneration"
  >,
  input: {
    context: TContext;
    hasStoredValues: boolean;
    storedValues: FeatureValues<TSchema> | undefined;
    lastGenerationSpec: unknown | undefined;
  },
): FeatureValues<TSchema> {
  const fromGeneration =
    input.lastGenerationSpec === undefined
      ? null
      : definition.fromGeneration(input.lastGenerationSpec, input.context);
  const seed =
    (input.hasStoredValues ? input.storedValues : undefined) ??
    fromGeneration ??
    definition.initialValues?.(input.context) ??
    definition.defaults;
  return definition.normalize(seed, input.context);
}

export function resolveFeatureFormValues<
  TSchema extends FeatureFieldSchema,
  TCreateBody,
  TContext,
>(
  definition: Pick<
    FeatureDefinition<TSchema, TCreateBody, TContext>,
    "defaults" | "initialValues" | "normalize" | "fromGeneration"
  >,
  input: {
    context: TContext;
    contextReady: boolean;
    storedValues: FeatureValues<TSchema> | undefined;
    generationsFetched: boolean;
    lastGenerationSpec: unknown | undefined;
  },
): FeatureValues<TSchema> {
  if (input.storedValues !== undefined) {
    return input.storedValues;
  }
  if (!input.contextReady) {
    return definition.defaults;
  }
  return resolveFeatureSeedValues(definition, {
    context: input.context,
    hasStoredValues: false,
    storedValues: undefined,
    lastGenerationSpec: input.generationsFetched
      ? input.lastGenerationSpec
      : undefined,
  });
}
