import type { components } from "../../../../generated/backend-openapi";
import {
  resolveVariantChoice,
  variantChoiceId,
  type CatalogSupportedModels,
  type CatalogVariantChoice,
} from "../../../lib/catalogVariantChoice.ts";
import {
  getIcLoraRecipe,
  icLoraImageValue,
  isIcLoraFieldHidden,
  isIcLoraImageRequirementKnown,
  missingIcLoraImage,
  type IcLoraRecipe,
  type IcLoraRecipeId,
} from "../../../../lib/ic-lora-recipes.ts";
import {
  withDefaultOffering,
  type OfferingId,
} from "../../../../lib/video-generation-model-specs.ts";
import { readGenerationParams } from "../../../lib/resultsFeedModel.ts";
import {
  applyFeatureFieldChange,
  type AssetRef,
  type FeatureDefinition,
  type FeatureFieldChange,
  type FeatureValues,
} from "../types.ts";
import { icLoraSelectedFps } from "./icLoraFps.ts";
import { normalizeIcLoraRecipeValues } from "./icLoraNormalize.ts";
import {
  icLoraModelOfferingIds,
  icLoraRecipeForm,
  IC_LORA_AUDIO_MODES,
  STRENGTH_MAX,
  STRENGTH_MIN,
  type IcLoraRecipeAudioMode,
} from "./icLoraRecipeForm.ts";
import {
  icLoraResolutionFor,
  isIcLoraResolution,
  type IcLoraRecipeResolution,
} from "./icLoraResolution.ts";
import { validateIcLoraRecipe } from "./icLoraValidate.ts";
import {
  hasAdvertisedCapability,
  readGenerationInputs,
  readPrompt,
  readVideoGenerationInput,
  resolveCapableModel,
  type VideoInputContext,
} from "./videoFeature.ts";

type CreateIcLoraRecipeRequest = components["schemas"]["CreateIcLoraRecipeRequest"];

export type IcLoraRecipeSchema = {
  video: { kind: "video-asset"; value: AssetRef | null };
  /** The reference still of a recipe with a `referenceImage`. Null for the others. */
  image: { kind: "image-asset"; value: AssetRef | null };
  prompt: { kind: "textarea" };
  loraStrength: { kind: "slider"; value: number };
  model: { kind: "options"; value: OfferingId };
  variant: { kind: "options"; value: string };
  resolution: { kind: "options"; value: IcLoraRecipeResolution };
  fps: { kind: "options"; value: number | null };
  audioMode: { kind: "options"; value: IcLoraRecipeAudioMode };
};

export type IcLoraRecipeValues = FeatureValues<IcLoraRecipeSchema>;
export type IcLoraRecipeContext = VideoInputContext & {
  catalogVariants?: CatalogVariantChoice[];
  /** The catalog `supported_models`. A model outside it is listed but disabled. */
  supportedModels?: CatalogSupportedModels;
  /** The packaged look image of a recipe with a `referenceImage`. */
  imageSeed?: AssetRef | null;
  /** The catalog `reference_image_required`. Undefined until the catalog loads. */
  referenceImageRequired?: boolean;
};

/** Fallback before the catalog loads. Catalog default_settings replaces it. */
export const IC_LORA_RECIPE_LORA_STRENGTH = 1;

export type IcLoraRecipeCatalogSettings = {
  lora_strength?: number | null;
  audio_mode?: string | null;
  /** The catalog entry's `allows_empty_prompt`. A blank prompt is then valid. */
  allows_empty_prompt?: boolean | null;
};

export function icLoraRecipeDefaults(
  settings?: IcLoraRecipeCatalogSettings | null,
): { loraStrength: number; audioMode: IcLoraRecipeAudioMode } {
  const strength = settings?.lora_strength;
  const audio = settings?.audio_mode;
  return {
    loraStrength: isLoraStrength(strength) ? strength : IC_LORA_RECIPE_LORA_STRENGTH,
    audioMode: isAudioMode(audio) ? audio : "source",
  };
}

function isLoraStrength(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isFinite(value) &&
    value >= STRENGTH_MIN &&
    value <= STRENGTH_MAX
  );
}

function isAudioMode(value: unknown): value is IcLoraRecipeAudioMode {
  return (
    typeof value === "string" &&
    (IC_LORA_AUDIO_MODES as readonly string[]).includes(value)
  );
}

function readStoredLora(
  params: Record<string, unknown>,
  fallback: number,
): { scale: number; variantId: string } {
  const loras = params.loras;
  const entry =
    Array.isArray(loras) && loras.length > 0
      ? (loras[0] as { scale?: unknown; variantId?: unknown })
      : null;
  const scale = isLoraStrength(entry?.scale)
    ? entry.scale
    : isLoraStrength(params.scale)
      ? params.scale
      : fallback;
  const variantId = typeof entry?.variantId === "string" ? entry.variantId : "";
  return { scale, variantId };
}

function defaultsFor(
  recipe: IcLoraRecipe,
  settings?: IcLoraRecipeCatalogSettings | null,
): IcLoraRecipeValues {
  const catalog = icLoraRecipeDefaults(settings);
  return {
    video: null,
    image: null,
    prompt: recipe.demoPrompt,
    loraStrength: catalog.loraStrength,
    model: "ltx-2.5-fast",
    variant: "",
    resolution: "720p",
    // null is Original: the closest supported rate that is not above the source.
    fps: null,
    audioMode: catalog.audioMode,
  };
}

export function createIcLoraRecipeDefinition(
  recipe: IcLoraRecipe,
  settings?: IcLoraRecipeCatalogSettings | null,
): FeatureDefinition<
  IcLoraRecipeSchema,
  CreateIcLoraRecipeRequest,
  IcLoraRecipeContext,
  IcLoraRecipeId
> {
  const defaults = defaultsFor(recipe, settings);
  const promptHidden = isIcLoraFieldHidden(recipe, "prompt");
  const audioHidden = isIcLoraFieldHidden(recipe, "audioMode");
  const strengthHidden = isIcLoraFieldHidden(recipe, "loraStrength");
  const promptOptional = promptHidden || settings?.allows_empty_prompt === true;
  const upscalesSource = recipe.upscalesSource === true;

  function seedValues(context: IcLoraRecipeContext): IcLoraRecipeValues {
    return {
      ...withDefaultOffering(context.specs, { ...defaults }),
      video: context.seed,
      image: icLoraImageValue(recipe, context.imageSeed),
    };
  }

  function fromGeneration(
    spec: unknown,
    context: IcLoraRecipeContext,
  ): IcLoraRecipeValues | null {
    const params = readGenerationParams(spec);
    const inputs = readVideoGenerationInput(spec);
    if (!params || !inputs) return null;
    const stored = readStoredLora(params, defaults.loraStrength);
    const savedImage = readGenerationInputs(spec, "image");
    return {
      video: inputs.video,
      // A saved `image: null` is a real choice. Only a missing key takes the seed.
      image: icLoraImageValue(
        recipe,
        savedImage != null ? savedImage.image : context.imageSeed,
      ),
      prompt: readPrompt(spec),
      loraStrength: stored.scale,
      variant: stored.variantId,
      model: resolveCapableModel(params.model, context.specs, "ic_lora", defaults.model),
      resolution: isIcLoraResolution(params.resolution) ? params.resolution : defaults.resolution,
      fps: icLoraSelectedFps(
        typeof params.fps === "number" && params.fps > 0 ? params.fps : null,
        context.videoFps,
      ),
      audioMode: isAudioMode(params.audioMode) ? params.audioMode : defaults.audioMode,
    };
  }

  function toCreateBody(
    values: IcLoraRecipeValues,
    context?: IcLoraRecipeContext,
  ): CreateIcLoraRecipeRequest {
    if (values.video == null) {
      throw new Error("Video is required.");
    }
    const variantId = variantChoiceId(
      resolveVariantChoice(
        context?.catalogVariants ?? [],
        values.variant,
        values.model,
        context ? icLoraModelOfferingIds(context) : [],
      ),
    );
    const image = icLoraImageValue(recipe, values.image);
    return {
      contract_version: 1,
      inputs: {
        video: { assetId: values.video.assetId },
        ...(image != null ? { image: { assetId: image.assetId } } : {}),
      },
      params: {
        prompt: promptHidden ? defaults.prompt : values.prompt.trim(),
        model: values.model,
        // Stored values skip normalize, so the source may no longer offer this cell.
        resolution: context
          ? icLoraResolutionFor(values.resolution, context, defaults.resolution, upscalesSource)
          : values.resolution,
        ...(values.fps != null ? { fps: values.fps } : {}),
        audioMode: audioHidden ? defaults.audioMode : values.audioMode,
        scale: strengthHidden ? defaults.loraStrength : values.loraStrength,
        ...(variantId ? { variantId } : {}),
        promptProvenance: "typed",
      },
    };
  }

  return {
    id: recipe.id,
    title: recipe.title,
    defaults,
    initialValues: seedValues,
    resetValues: seedValues,
    form: (values, context) => icLoraRecipeForm(recipe, values, context),
    applyChange: (values, change: FeatureFieldChange<IcLoraRecipeSchema>) =>
      applyFeatureFieldChange(values, change),
    normalize: (values, context) =>
      normalizeIcLoraRecipeValues(values, context, defaults, upscalesSource),
    fromGeneration,
    validate: (values, context) =>
      validateIcLoraRecipe(recipe, values, context, promptOptional),
    toCreateBody,
    // The size must be known: the resolution cells depend on the source size.
    isReady: (values, context) =>
      values.video != null &&
      context.videoWidth != null &&
      context.videoHeight != null &&
      isIcLoraImageRequirementKnown(recipe, context.referenceImageRequired) &&
      missingIcLoraImage(recipe, values.image, context.referenceImageRequired) == null,
    isUnavailable: (_values, context) => !hasAdvertisedCapability(context.specs, "ic_lora"),
  };
}

export function icLoraRecipeDefinition(id: IcLoraRecipeId) {
  return createIcLoraRecipeDefinition(getIcLoraRecipe(id));
}
