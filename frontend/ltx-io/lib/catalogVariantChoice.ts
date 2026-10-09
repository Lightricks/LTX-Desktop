export type CatalogVariantChoice = {
  id: string;
  label: string;
  baseModel: string;
  downloaded: boolean;
};

export type VariantKind = "IC-LoRA" | "LoRA";

export type VariantOption = {
  value: string;
  label: string;
  warning?: string;
};

/** One read of the installed checkpoints. Dropdown, stored id, and warnings use it. */
export type VariantChoice =
  | { mode: "unloaded"; variantId: string }
  | { mode: "none" }
  | { mode: "one"; variant: CatalogVariantChoice }
  | { mode: "follow"; variant: CatalogVariantChoice | null }
  | { mode: "picker"; options: CatalogVariantChoice[]; selected: CatalogVariantChoice };

type CatalogVariantRecord = {
  id: string;
  label: string;
  base_model: string;
};

export function offeringFamily(offeringId: string): string {
  const match = /^ltx-(\d+(?:\.\d+)?)/i.exec(offeringId);
  if (match == null) return offeringId;
  return `LTX-${match[1]}`;
}

/** LTX-2 weights (camera LoRAs) run on both current families, so they never mismatch. */
const ALL_FAMILIES_BASE_MODEL = "LTX-2";

function runsOnFamily(baseModel: string, family: string): boolean {
  return baseModel === family || baseModel === ALL_FAMILIES_BASE_MODEL;
}

function trainedOnWarning(
  baseModel: string,
  offeringId: string,
  kind: VariantKind,
): string | undefined {
  const family = offeringFamily(offeringId);
  if (runsOnFamily(baseModel, family)) return undefined;
  return `This ${kind} Variant was trained on ${baseModel}. Results with ${family} may not be optimal.`;
}

function onlyInstalledWarning(label: string, offeringId: string): string {
  return `You only have the ${label} Variant. Results with ${offeringFamily(offeringId)} may not be optimal.`;
}

function followsSelectedModel(
  installed: readonly CatalogVariantChoice[],
  offeringIds: readonly string[],
): boolean {
  const families = [...new Set(offeringIds.map(offeringFamily))];
  return (
    families.length >= 2 &&
    families.every((family) => installed.some((variant) => variant.baseModel === family))
  );
}

export function resolveVariantChoice(
  variants: readonly CatalogVariantChoice[] | undefined,
  selectedId: string,
  offeringId: string,
  offeringIds: readonly string[],
): VariantChoice {
  if (variants == null || variants.length === 0) {
    return { mode: "unloaded", variantId: selectedId };
  }
  const installed = variants.filter((variant) => variant.downloaded);
  if (followsSelectedModel(installed, offeringIds)) {
    const family = offeringFamily(offeringId);
    return {
      mode: "follow",
      variant: installed.find((variant) => variant.baseModel === family) ?? null,
    };
  }
  if (installed.length === 0) return { mode: "none" };
  if (installed.length === 1) {
    const variant = installed[0];
    if (variant == null) return { mode: "none" };
    return { mode: "one", variant };
  }
  const family = offeringFamily(offeringId);
  const selected =
    installed.find((variant) => variant.id === selectedId) ??
    installed.find((variant) => variant.baseModel === family) ??
    installed[0];
  if (selected == null) return { mode: "none" };
  return { mode: "picker", options: installed, selected };
}

export function variantChoiceId(choice: VariantChoice): string {
  switch (choice.mode) {
    case "unloaded":
      return choice.variantId;
    case "none":
      return "";
    case "one":
      return choice.variant.id;
    case "follow":
      return choice.variant?.id ?? "";
    case "picker":
      return choice.selected.id;
    default: {
      const unreachable: never = choice;
      return unreachable;
    }
  }
}

export function variantChoiceWarning(
  choice: VariantChoice,
  offeringId: string,
  kind: VariantKind,
): string | undefined {
  switch (choice.mode) {
    case "one":
      return runsOnFamily(choice.variant.baseModel, offeringFamily(offeringId))
        ? undefined
        : onlyInstalledWarning(choice.variant.label, offeringId);
    case "picker":
      return trainedOnWarning(choice.selected.baseModel, offeringId, kind);
    case "unloaded":
    case "none":
    case "follow":
      return undefined;
    default: {
      const unreachable: never = choice;
      return unreachable;
    }
  }
}

export function variantChoiceOptions(
  choice: VariantChoice,
  offeringId: string,
  kind: VariantKind,
): VariantOption[] | null {
  if (choice.mode !== "picker") return null;
  return choice.options.map((variant) => ({
    value: variant.id,
    label: variant.label,
    warning: trainedOnWarning(variant.baseModel, offeringId, kind),
  }));
}

/**
 * The catalog `supported_models` families. Unlike the variant warning, the backend
 * rejects a family that is not listed. An unloaded catalog (undefined) blocks nothing.
 */
export type CatalogSupportedModels = readonly string[] | null | undefined;

function isFamilySupported(supported: CatalogSupportedModels, offeringId: string): boolean {
  if (supported == null || supported.length === 0) return true;
  return supported.includes(offeringFamily(offeringId));
}

/** Why this model cannot run the LoRA, or undefined when it can. */
export function unsupportedModelReason(
  supported: CatalogSupportedModels,
  offeringId: string,
  kind: VariantKind,
): string | undefined {
  if (isFamilySupported(supported, offeringId)) return undefined;
  return `This ${kind} runs on ${(supported ?? []).join(" or ")} only.`;
}

/** Keeps a supported model. Otherwise picks the first supported one in the list. */
export function resolveSupportedModel<TModel extends string>(
  model: TModel,
  offeringIds: readonly TModel[],
  supported: CatalogSupportedModels,
): TModel {
  if (isFamilySupported(supported, model)) return model;
  return offeringIds.find((id) => isFamilySupported(supported, id)) ?? model;
}

export function catalogVariantsFor(
  item:
    | { download: { variants: readonly CatalogVariantRecord[] } }
    | null
    | undefined,
  downloadedIds: readonly string[] | null | undefined,
): CatalogVariantChoice[] {
  if (item == null) return [];
  const downloaded = new Set(downloadedIds ?? []);
  return item.download.variants.map((variant) => ({
    id: variant.id,
    label: variant.label,
    baseModel: variant.base_model,
    downloaded: downloaded.has(variant.id),
  }));
}
