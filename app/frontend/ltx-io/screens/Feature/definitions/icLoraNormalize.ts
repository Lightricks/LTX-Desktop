import {
  resolveSupportedModel,
  resolveVariantChoice,
  variantChoiceId,
} from "../../../lib/catalogVariantChoice.ts";
import { icLoraSelectedFps } from "./icLoraFps.ts";
import { icLoraModelOfferingIds } from "./icLoraRecipeForm.ts";
import { isTileBlocked, isWindowBlocked } from "./icLoraTileOptions.ts";
import { AUTO_TILE, windowFrameChoices } from "./icLoraTilingControls.ts";
import { icLoraResolutionFor, type UpscalesSource } from "./icLoraResolution.ts";
import type { IcLoraRecipeContext, IcLoraRecipeValues } from "./icLoraRecipe.ts";
import { capabilityModelOptions, resolveCapableModel } from "./videoFeature.ts";

type ModelVariantValues = Pick<IcLoraRecipeValues, "model" | "variant">;
type ModelVariantContext = Pick<
  IcLoraRecipeContext,
  "specs" | "catalogVariants" | "supportedModels"
>;

/**
 * The model the specs and the catalog allow, and the variant that fits it.
 * Both IC-LoRA forms (video and image list) use it, so a fix lands in both.
 */
export function normalizeIcLoraModelAndVariant(
  values: ModelVariantValues,
  context: ModelVariantContext,
  defaultModel: ModelVariantValues["model"],
): ModelVariantValues {
  const model = resolveSupportedModel(
    resolveCapableModel(values.model, context.specs, "ic_lora", defaultModel),
    capabilityModelOptions(context.specs, "ic_lora").map((option) => option.value),
    context.supportedModels,
  );
  const variant = variantChoiceId(
    resolveVariantChoice(
      context.catalogVariants ?? [],
      values.variant ?? "",
      model,
      icLoraModelOfferingIds(context),
    ),
  );
  return { model, variant };
}

/** The variant id for a create body, or "" when the catalog has no variant to send. */
export function icLoraVariantId(
  values: ModelVariantValues,
  context?: ModelVariantContext,
): string {
  return variantChoiceId(
    resolveVariantChoice(
      context?.catalogVariants ?? [],
      values.variant,
      values.model,
      context ? icLoraModelOfferingIds(context) : [],
    ),
  );
}

/**
 * Brings stored values in line with the loaded specs, the catalog and the source
 * clip. It returns the same object when nothing changes.
 */
export function normalizeIcLoraRecipeValues(
  values: IcLoraRecipeValues,
  context: IcLoraRecipeContext,
  defaults: Pick<IcLoraRecipeValues, "model" | "resolution">,
  upscalesSource: UpscalesSource = false,
): IcLoraRecipeValues {
  const { model, variant } = normalizeIcLoraModelAndVariant(values, context, defaults.model);
  const resolution = icLoraResolutionFor(
    values.resolution,
    context,
    defaults.resolution,
    upscalesSource,
  );
  const fps = icLoraSelectedFps(values.fps, context.videoFps);
  // The budget refuses the stored tile size at this cell: fall back to Auto.
  const tilingSize =
    values.tilingSize !== AUTO_TILE && isTileBlocked(values.tilingSize, context)
      ? AUTO_TILE
      : values.tilingSize;
  // Auto is stored as null. A maximum falls back to Auto when it is not offered or the preflight
  // refuses it. A carry stays as the user set it: a refused carry is disabled in the list, and
  // Generate shows its refusal.
  const maxWindowFrames =
    values.maxWindowFrames != null &&
    (!windowFrameChoices().includes(values.maxWindowFrames) ||
      isWindowBlocked(values.maxWindowFrames, context))
      ? null
      : values.maxWindowFrames;
  if (
    model === values.model &&
    tilingSize === values.tilingSize &&
    maxWindowFrames === values.maxWindowFrames &&
    resolution === values.resolution &&
    fps === values.fps &&
    variant === values.variant
  ) {
    return values;
  }
  return { ...values, model, resolution, fps, variant, tilingSize, maxWindowFrames };
}
