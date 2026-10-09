import {
  resolveSupportedModel,
  resolveVariantChoice,
  variantChoiceId,
} from "../../../lib/catalogVariantChoice.ts";
import { icLoraSelectedFps } from "./icLoraFps.ts";
import { icLoraModelOfferingIds } from "./icLoraRecipeForm.ts";
import { icLoraResolutionFor } from "./icLoraResolution.ts";
import type { IcLoraRecipeContext, IcLoraRecipeValues } from "./icLoraRecipe.ts";
import { capabilityModelOptions, resolveCapableModel } from "./videoFeature.ts";

/**
 * Brings stored values in line with the loaded specs, the catalog and the source
 * clip. It returns the same object when nothing changes.
 */
export function normalizeIcLoraRecipeValues(
  values: IcLoraRecipeValues,
  context: IcLoraRecipeContext,
  defaults: Pick<IcLoraRecipeValues, "model" | "resolution">,
  upscalesSource = false,
): IcLoraRecipeValues {
  const model = resolveSupportedModel(
    resolveCapableModel(values.model, context.specs, "ic_lora", defaults.model),
    capabilityModelOptions(context.specs, "ic_lora").map((option) => option.value),
    context.supportedModels,
  );
  const resolution = icLoraResolutionFor(
    values.resolution,
    context,
    defaults.resolution,
    upscalesSource,
  );
  const fps = icLoraSelectedFps(values.fps, context.videoFps);
  const variant = variantChoiceId(
    resolveVariantChoice(
      context.catalogVariants ?? [],
      values.variant ?? "",
      model,
      icLoraModelOfferingIds(context),
    ),
  );
  if (
    model === values.model &&
    resolution === values.resolution &&
    fps === values.fps &&
    variant === values.variant
  ) {
    return values;
  }
  return { ...values, model, resolution, fps, variant };
}
