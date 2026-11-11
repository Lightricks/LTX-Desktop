import type { BudgetReadout } from "@/lib/budget-figure";
import type { MemoryFitsModel, MemoryMeterModel, OptionMemory } from "@/lib/memory-meter";
import type { HomeFeatureId } from "../../../lib/home-features";
import type { MotionPath } from "../../lib/motionPathSampler";

export type FieldOption<TValue extends string | number = string | number> = {
  value: TValue;
  label: string;
  /** Soft caveat or budget-risk line shown on hover. */
  warning?: string;
  /** Gray quality caveat, or yellow budget risk. Omitted means caveat. */
  warningTone?: "caveat" | "risk";
  /** The option is listed but cannot be picked. The text says why. */
  disabledReason?: string;
  /** A line under the label in the picker list. `disabledReason` replaces it. */
  caption?: string;
  /** A picture next to the label, in the list and in the picker box. */
  thumbnail?: string;
  /** Structured need/have pair. Present only for a budget limit. */
  budget?: BudgetReadout | null;
  /** A caveat the budget badge carries in its tooltip, because the badge replaces the warning icon. */
  budgetNote?: string;
  /** This option's memory ask with the other choices kept: the row label and the hover preview. */
  memory?: OptionMemory;
};

export type AssetRef = {
  assetId: string;
};

export function toDurableAssetRef(asset: { id: string }): AssetRef {
  return { assetId: asset.id };
}

/** One still on the timeline of a `keyframes` field. The `id` is a React key and is never sent. */
export type FeatureKeyframe = {
  id: string;
  asset: AssetRef;
  frameIndex: number;
  /** 0 holds nothing and 1 holds the image fully. */
  strength: number;
};

export type FeatureFieldKind =
  | "textarea"
  | "options"
  | "image-asset"
  | "image-assets"
  | "keyframes"
  | "slider"
  | "audio-asset"
  | "video-asset"
  | "motion-path";

export type FeatureFieldSpec =
  | { kind: "textarea"; value?: never }
  | { kind: "options"; value: string | number | null }
  | { kind: "image-asset"; value: AssetRef | null }
  | { kind: "image-assets"; value: AssetRef[] }
  | { kind: "keyframes"; value: FeatureKeyframe[] }
  | { kind: "slider"; value: number }
  | { kind: "audio-asset"; value: AssetRef | null }
  | { kind: "video-asset"; value: AssetRef | null }
  /** The tracks of a path field. The list is empty until the user draws one. */
  | { kind: "motion-path"; value: MotionPath[] };

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

export type ImageAssetsFieldKey<TSchema extends FeatureFieldSchema> =
  FieldKeyOfKind<TSchema, "image-assets">;

export type KeyframesFieldKey<TSchema extends FeatureFieldSchema> =
  FieldKeyOfKind<TSchema, "keyframes">;

export type SliderFieldKey<TSchema extends FeatureFieldSchema> =
  FieldKeyOfKind<TSchema, "slider">;

export type AudioAssetFieldKey<TSchema extends FeatureFieldSchema> =
  FieldKeyOfKind<TSchema, "audio-asset">;

export type VideoAssetFieldKey<TSchema extends FeatureFieldSchema> =
  FieldKeyOfKind<TSchema, "video-asset">;

export type MotionPathFieldKey<TSchema extends FeatureFieldSchema> =
  FieldKeyOfKind<TSchema, "motion-path">;

type FeatureFieldBase<TKey extends string> = {
  id: string;
  label: string;
  dataKey: TKey;
  /** Help text shown on hover next to the label. */
  tooltip?: string;
};

export type TextareaFieldDefForKey<
  TSchema extends FeatureFieldSchema,
  TKey extends TextareaFieldKey<TSchema>,
> = FeatureFieldBase<TKey> & {
  kind: "textarea";
  placeholder?: string;
  helpText?: string;
  disabledReason?: string | null;
  compact?: boolean;
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

export type ImageAssetsFieldDefForKey<
  TSchema extends FeatureFieldSchema,
  TKey extends ImageAssetsFieldKey<TSchema>,
> = FeatureFieldBase<TKey> & {
  kind: "image-assets";
  /** The most images the field takes. */
  maxCount: number;
  /** One line under the label, shown while the list is empty and after it fills. */
  hint?: string;
};

export type ImageAssetsFieldDef<TSchema extends FeatureFieldSchema> = {
  [TKey in ImageAssetsFieldKey<TSchema>]: ImageAssetsFieldDefForKey<
    TSchema,
    TKey
  >;
}[ImageAssetsFieldKey<TSchema>];

export type KeyframesFieldDefForKey<
  TSchema extends FeatureFieldSchema,
  TKey extends KeyframesFieldKey<TSchema>,
> = FeatureFieldBase<TKey> & {
  kind: "keyframes";
  /** The most stills the field takes. */
  maxCount: number;
  /**
   * The timeline is the clip length and frame rate, which other fields hold. The field reads
   * them but never writes them. A change of either one re-times the stills in the definition.
   */
  durationKey: OptionsFieldKey<TSchema>;
  fpsKey: OptionsFieldKey<TSchema>;
};

export type KeyframesFieldDef<TSchema extends FeatureFieldSchema> = {
  [TKey in KeyframesFieldKey<TSchema>]: KeyframesFieldDefForKey<TSchema, TKey>;
}[KeyframesFieldKey<TSchema>];

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

export type MotionPathFieldDefForKey<
  TSchema extends FeatureFieldSchema,
  TKey extends MotionPathFieldKey<TSchema>,
> = FeatureFieldBase<TKey> & {
  kind: "motion-path";
  /**
   * The one field kind that writes other fields. The tile holds the image and the editor
   * holds the clip length, so a `MotionPathFieldEdit` can change `imageKey` or `durationKey`
   * as well as this field.
   */
  imageKey: ImageAssetFieldKey<TSchema>;
  durationKey: OptionsFieldKey<TSchema>;
  durationOptions: FieldOption<number>[];
};

/** What the path tile reports: a new image, or the accepted tracks with their clip length. */
export type MotionPathFieldEdit =
  | { image: AssetRef | null }
  | { paths: MotionPath[]; duration: number };

export type MotionPathFieldDef<TSchema extends FeatureFieldSchema> = {
  [TKey in MotionPathFieldKey<TSchema>]: MotionPathFieldDefForKey<TSchema, TKey>;
}[MotionPathFieldKey<TSchema>];

export type FeatureFormField<TSchema extends FeatureFieldSchema> =
  | TextareaFieldDef<TSchema>
  | OptionsFieldDef<TSchema>
  | ImageAssetFieldDef<TSchema>
  | ImageAssetsFieldDef<TSchema>
  | KeyframesFieldDef<TSchema>
  | SliderFieldDef<TSchema>
  | AudioAssetFieldDef<TSchema>
  | VideoAssetFieldDef<TSchema>
  | MotionPathFieldDef<TSchema>;

/**
 * The fields the Advanced section renders. Every one is an options field whose first option is
 * Auto and whose `defaultValue` is that option, so "Reset to Auto" has one value to set.
 */
export type AdvancedFormField<TSchema extends FeatureFieldSchema> = OptionsFieldDef<TSchema> & {
  defaultValue: NonNullable<OptionsFieldDef<TSchema>["defaultValue"]>;
};

export type FeatureFormModel<TSchema extends FeatureFieldSchema> = {
  fields: FeatureFormField<TSchema>[];
  /** Options fields for a collapsed Advanced section. No section when empty. */
  advancedFields?: AdvancedFormField<TSchema>[];
  /** The memory meter under the settings. Features without a priced cell leave it out. */
  memory?: MemoryMeterModel | null;
  /** Before the input that sets the number is picked: which sizes fit. Shown when `memory` is not. */
  memoryFits?: MemoryFitsModel | null;
  /** Estimated generation time shown under Generate when `memory` is absent. */
  estimatedTimeText?: string | null;
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
  TKey extends ImageAssetsFieldKey<TSchema>,
>(change: {
  kind: "image-assets";
  fieldId: string;
  dataKey: TKey;
  value: AssetRef[];
}): FeatureFieldChange<TSchema>;
export function createFeatureFieldChange<
  TSchema extends FeatureFieldSchema,
  TKey extends KeyframesFieldKey<TSchema>,
>(change: {
  kind: "keyframes";
  fieldId: string;
  dataKey: TKey;
  value: FeatureKeyframe[];
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
  TKey extends MotionPathFieldKey<TSchema>,
>(change: {
  kind: "motion-path";
  fieldId: string;
  dataKey: TKey;
  value: MotionPath[];
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

export function readImageAssetsFieldValue<
  TSchema extends FeatureFieldSchema,
  TKey extends ImageAssetsFieldKey<TSchema>,
>(values: FeatureValues<TSchema>, dataKey: TKey): AssetRef[] {
  return readConstrainedFieldValue<TSchema, TKey, AssetRef[]>(values, dataKey);
}

export function readKeyframesFieldValue<
  TSchema extends FeatureFieldSchema,
  TKey extends KeyframesFieldKey<TSchema>,
>(values: FeatureValues<TSchema>, dataKey: TKey): FeatureKeyframe[] {
  return readConstrainedFieldValue<TSchema, TKey, FeatureKeyframe[]>(
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

export function readMotionPathFieldValue<
  TSchema extends FeatureFieldSchema,
  TKey extends MotionPathFieldKey<TSchema>,
>(values: FeatureValues<TSchema>, dataKey: TKey): MotionPath[] {
  return readConstrainedFieldValue<TSchema, TKey, MotionPath[]>(values, dataKey);
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
  /** Budget risk renders yellow and does not use the error color. */
  tone?: "risk";
  /** Set on budget-risk issues. Drives the generate hint and the T3 confirm. */
  tier?: "tight" | "over" | "unmeasured";
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
