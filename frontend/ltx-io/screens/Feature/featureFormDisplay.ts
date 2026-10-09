import type {
  AudioAssetFieldDef,
  FeatureFieldSchema,
  FeatureFormField,
  ImageAssetFieldDef,
  OptionsFieldDef,
  SliderFieldDef,
  TextareaFieldDef,
  VideoAssetFieldDef,
} from "./types.ts";

export type FeatureFormDisplayGroup<TSchema extends FeatureFieldSchema> =
  | { kind: "textarea"; field: TextareaFieldDef<TSchema> }
  | { kind: "image-asset"; field: ImageAssetFieldDef<TSchema> }
  | { kind: "slider"; field: SliderFieldDef<TSchema> }
  | { kind: "audio-asset"; field: AudioAssetFieldDef<TSchema> }
  | { kind: "video-asset"; field: VideoAssetFieldDef<TSchema> }
  | { kind: "options"; fields: OptionsFieldDef<TSchema>[] };

function assertNever(value: never): never {
  throw new Error(`Unhandled feature form field: ${String(value)}`);
}

function isIdPermutation(
  originalIds: readonly string[],
  orderedIds: readonly string[],
): boolean {
  if (originalIds.length !== orderedIds.length) return false;
  if (new Set(originalIds).size !== originalIds.length) return false;
  if (new Set(orderedIds).size !== orderedIds.length) return false;
  const originalSet = new Set(originalIds);
  return orderedIds.every((id) => originalSet.has(id));
}

export function applyOptionFieldOrder<TSchema extends FeatureFieldSchema>(
  fields: readonly FeatureFormField<TSchema>[],
  orderOptionFields: (
    fields: readonly OptionsFieldDef<TSchema>[],
  ) => readonly OptionsFieldDef<TSchema>[],
): readonly FeatureFormField<TSchema>[] {
  const originalOptions = fields.filter(
    (field): field is OptionsFieldDef<TSchema> => field.kind === "options",
  );
  const orderedOptions = [...orderOptionFields(originalOptions)];
  if (
    !isIdPermutation(
      originalOptions.map((field) => field.id),
      orderedOptions.map((field) => field.id),
    )
  ) {
    return fields;
  }

  let optionIndex = 0;
  return fields.map((field): FeatureFormField<TSchema> => {
    switch (field.kind) {
      case "textarea":
      case "image-asset":
      case "slider":
      case "audio-asset":
      case "video-asset":
        return field;
      case "options":
        return orderedOptions[optionIndex++] ?? field;
      default:
        return assertNever(field);
    }
  });
}

export function groupConsecutiveOptionFields<TSchema extends FeatureFieldSchema>(
  fields: readonly FeatureFormField<TSchema>[],
): FeatureFormDisplayGroup<TSchema>[] {
  const groups: FeatureFormDisplayGroup<TSchema>[] = [];
  for (const field of fields) {
    switch (field.kind) {
      case "textarea":
        groups.push({ kind: "textarea", field });
        break;
      case "image-asset":
        groups.push({ kind: "image-asset", field });
        break;
      case "slider":
        groups.push({ kind: "slider", field });
        break;
      case "audio-asset":
        groups.push({ kind: "audio-asset", field });
        break;
      case "video-asset":
        groups.push({ kind: "video-asset", field });
        break;
      case "options": {
        const last = groups.at(-1);
        if (last?.kind === "options") {
          last.fields.push(field);
        } else {
          groups.push({ kind: "options", fields: [field] });
        }
        break;
      }
      default:
        return assertNever(field);
    }
  }
  return groups;
}

/**
 * The value an options control shows. A stored `null` means the field is unset,
 * so the control shows `defaultValue`, or the first option when there is none.
 */
export function optionControlValue<TValue extends string | number>(
  stored: TValue | null,
  options: readonly { value: TValue }[],
  defaultValue?: TValue,
): TValue {
  if (stored != null) return stored;
  if (defaultValue !== undefined) return defaultValue;
  const first = options[0];
  if (first == null) {
    throw new Error("options field has no choices");
  }
  return first.value;
}
