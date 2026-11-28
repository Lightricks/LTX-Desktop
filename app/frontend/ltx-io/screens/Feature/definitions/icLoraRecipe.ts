import type { components } from "../../../../generated/backend-openapi";
import {
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
  type IcLoraVideoRecipeId,
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
import {
  IC_LORA_CONTROL_TYPES,
  isIcLoraControlType,
  type IcLoraControlType,
} from "../../../lib/icLoraControlType.ts";
import { icLoraSelectedFps } from "./icLoraFps.ts";
import {
  AUTO_NUMBER,
  AUTO_TILE,
  isTileChoice,
  overlapChoices,
  type IcLoraTileChoice,
} from "./icLoraTilingControls.ts";
import type { IcLoraRecipeMemory } from "./icLoraMemory.ts";
import { icLoraVariantId, normalizeIcLoraRecipeValues } from "./icLoraNormalize.ts";
import {
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
import { swapDemoNegative } from "./negativePrompt.ts";
import {
  hasAdvertisedCapability,
  readGenerationInputs,
  readNegativePrompt,
  readPrompt,
  readVideoGenerationInput,
  resolveCapableModel,
  type VideoInputContext,
} from "./videoFeature.ts";

type CreateIcLoraRecipeRequest = components["schemas"]["CreateIcLoraRecipeRequest"];
type IcLoraTilePreflightResponse = components["schemas"]["IcLoraTilePreflightResponse"];

export type IcLoraRecipeSchema = {
  video: { kind: "video-asset"; value: AssetRef | null };
  /** The reference still of a recipe with a `referenceImage`. Null for the others. */
  image: { kind: "image-asset"; value: AssetRef | null };
  prompt: { kind: "textarea" };
  negativePrompt: { kind: "textarea" };
  loraStrength: { kind: "slider"; value: number };
  model: { kind: "options"; value: OfferingId };
  variant: { kind: "options"; value: string };
  resolution: { kind: "options"; value: IcLoraRecipeResolution };
  fps: { kind: "options"; value: number | null };
  audioMode: { kind: "options"; value: IcLoraRecipeAudioMode };
  /** How the guide clip is read. Only a recipe with `controlMode` shows and sends it. */
  controlType: { kind: "options"; value: IcLoraControlType };
  /**
   * The Advanced controls. Only a recipe with `tilingControls` shows and sends them. Null is Auto
   * for the three numeric ones, and the backend picks the value. The tile size stores "auto".
   *
   * The share of a tile that the next tile covers.
   */
  tilingOverlap: { kind: "options"; value: number | null };
  /** Frames carried from one chunk to the next. */
  carryFrames: { kind: "options"; value: number | null };
  /** The tile the model works on. Auto keeps the catalog tile. */
  tilingSize: { kind: "options"; value: IcLoraTileChoice };
  /** Frames in one chunk of a long clip. */
  maxWindowFrames: { kind: "options"; value: number | null };
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
  /** The cells above 1080p this recipe offers. Empty without the `highResVideoToVideo` flag. */
  highResCells?: readonly ("1440p" | "2160p")[];
  /** The pictures of the control modes. The screen owns the assets. */
  controlThumbnails?: Record<IcLoraControlType, string>;
  /** The catalog entry picks its chunk window itself (`adaptive_window`). The Advanced section needs it. */
  tilingAdaptive?: boolean;
  /** The catalog tile, long side by short side. The Standard tile size is this. */
  tilingTile?: { long: number; short: number } | null;
  /** Verdicts for each tile, chunk and carry, plus the defaults a run uses. Null until they load. */
  tilePreflight?: IcLoraTilePreflightResponse | null;
  /** The catalog entry keeps the IC-LoRA on stage 2. The form then offers cells up to 2x the source. */
  twoStage?: boolean;
  /** Memory advice for the current cell and clip. Undefined or null while unknown. */
  recipeMemory?: IcLoraRecipeMemory | null;
};

/** Fallback before the catalog loads. Catalog default_settings replaces it. */
export const IC_LORA_RECIPE_LORA_STRENGTH = 1;

export type IcLoraRecipeCatalogSettings = {
  lora_strength?: number | null;
  audio_mode?: string | null;
  /** The catalog entry's `allows_empty_prompt`. A blank prompt is then valid. */
  allows_empty_prompt?: boolean | null;
  /** Generic `default_negative_prompt` from the catalog entry, when present. */
  default_negative_prompt?: string | null;
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

export function isLoraStrength(value: unknown): value is number {
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

export function readStoredLora(
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

/** A stored overlap that the control offers, else null (Auto). */
function readTilingOverlap(value: unknown): number | null {
  return typeof value === "number" && overlapChoices().includes(value) ? value : null;
}

/**
 * The overrides the user chose. An Auto control sends nothing, and the backend picks its value.
 * A choice is sent as it is: the body never depends on the advisory preflight.
 */
export function icLoraTilingParams(values: IcLoraRecipeValues) {
  return {
    ...(values.tilingOverlap != null ? { tilingOverlap: values.tilingOverlap } : {}),
    ...(values.carryFrames != null ? { carryFrames: values.carryFrames } : {}),
    ...(values.tilingSize !== AUTO_TILE ? { tilingSize: values.tilingSize } : {}),
    ...(values.maxWindowFrames != null ? { maxWindowFrames: values.maxWindowFrames } : {}),
  };
}

/** Auto is the value of an option, and the form stores it as null. */
function withAutoAsNull(values: IcLoraRecipeValues): IcLoraRecipeValues {
  const isAuto = (value: number | null) => value === AUTO_NUMBER;
  if (
    !isAuto(values.tilingOverlap) &&
    !isAuto(values.carryFrames) &&
    !isAuto(values.maxWindowFrames)
  ) {
    return values;
  }
  return {
    ...values,
    tilingOverlap: isAuto(values.tilingOverlap) ? null : values.tilingOverlap,
    carryFrames: isAuto(values.carryFrames) ? null : values.carryFrames,
    maxWindowFrames: isAuto(values.maxWindowFrames) ? null : values.maxWindowFrames,
  };
}

function defaultsFor(
  recipe: IcLoraRecipe,
  settings?: IcLoraRecipeCatalogSettings | null,
): IcLoraRecipeValues {
  const catalog = icLoraRecipeDefaults(settings);
  const promptHidden = isIcLoraFieldHidden(recipe, "prompt");
  return {
    video: null,
    image: null,
    prompt: recipe.demoPrompt,
    negativePrompt: promptHidden
      ? ""
      : recipe.demoNegativePrompt || settings?.default_negative_prompt || "",
    loraStrength: catalog.loraStrength,
    model: "ltx-2.5-fast",
    variant: "",
    // A recipe that lists no cell below the source starts at its lowest cell. The form then
    // normalizes it to the smallest offered cell, which is the Original cell.
    resolution: recipe.upscalesSource === "source-and-above" ? "270p" : "720p",
    // null is Original: the closest supported rate that is not above the source.
    fps: null,
    audioMode: catalog.audioMode,
    controlType: IC_LORA_CONTROL_TYPES[0],
    tilingOverlap: null,
    carryFrames: null,
    tilingSize: AUTO_TILE,
    maxWindowFrames: null,
  };
}

export function createIcLoraRecipeDefinition(
  recipe: IcLoraRecipe,
  settings?: IcLoraRecipeCatalogSettings | null,
): FeatureDefinition<
  IcLoraRecipeSchema,
  CreateIcLoraRecipeRequest,
  IcLoraRecipeContext,
  IcLoraVideoRecipeId
> {
  const defaults = defaultsFor(recipe, settings);
  const promptHidden = isIcLoraFieldHidden(recipe, "prompt");
  const audioHidden = isIcLoraFieldHidden(recipe, "audioMode");
  const strengthHidden = isIcLoraFieldHidden(recipe, "loraStrength");
  const promptOptional = promptHidden || settings?.allows_empty_prompt === true;
  const upscalesSource = recipe.upscalesSource ?? false;

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
      negativePrompt: readNegativePrompt(spec),
      loraStrength: stored.scale,
      variant: stored.variantId,
      model: resolveCapableModel(params.model, context.specs, "ic_lora", defaults.model),
      resolution: isIcLoraResolution(params.resolution) ? params.resolution : defaults.resolution,
      fps: icLoraSelectedFps(
        typeof params.fps === "number" && params.fps > 0 ? params.fps : null,
        context.videoFps,
      ),
      audioMode: isAudioMode(params.audioMode) ? params.audioMode : defaults.audioMode,
      controlType: isIcLoraControlType(params.controlType) ? params.controlType : defaults.controlType,
      tilingOverlap: readTilingOverlap(params.tilingOverlap),
      carryFrames: typeof params.carryFrames === "number" ? params.carryFrames : null,
      tilingSize: isTileChoice(params.tilingSize) ? params.tilingSize : AUTO_TILE,
      maxWindowFrames: typeof params.maxWindowFrames === "number" ? params.maxWindowFrames : null,
    };
  }

  function toCreateBody(
    values: IcLoraRecipeValues,
    context?: IcLoraRecipeContext,
  ): CreateIcLoraRecipeRequest {
    if (values.video == null) {
      throw new Error("Video is required.");
    }
    const variantId = icLoraVariantId(values, context);
    const image = icLoraImageValue(recipe, values.image);
    return {
      contract_version: 1,
      inputs: {
        video: { assetId: values.video.assetId },
        ...(image != null ? { image: { assetId: image.assetId } } : {}),
      },
      params: {
        prompt: promptHidden ? defaults.prompt : values.prompt.trim(),
        negativePrompt: promptHidden ? "" : values.negativePrompt.trim(),
        model: values.model,
        // Stored values skip normalize, so the source may no longer offer this cell.
        resolution: context
          ? icLoraResolutionFor(values.resolution, context, defaults.resolution, upscalesSource)
          : values.resolution,
        ...(values.fps != null ? { fps: values.fps } : {}),
        audioMode: audioHidden ? defaults.audioMode : values.audioMode,
        scale: strengthHidden ? defaults.loraStrength : values.loraStrength,
        ...(recipe.controlMode ? { controlType: values.controlType } : {}),
        ...(recipe.tilingControls ? icLoraTilingParams(values) : {}),
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
    applyChange: (values, change: FeatureFieldChange<IcLoraRecipeSchema>) => {
      const next = swapDemoNegative(values, applyFeatureFieldChange(values, change), {
        demoNegative: recipe.demoNegativePrompt,
        catalogDefault: promptHidden ? null : settings?.default_negative_prompt,
        changed: (before, after) =>
          (change.dataKey === "video" || change.dataKey === "prompt") &&
          (after.video?.assetId !== before.video?.assetId || after.prompt !== before.prompt),
      });
      return withAutoAsNull(next);
    },
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

export function icLoraRecipeDefinition(id: IcLoraVideoRecipeId) {
  return createIcLoraRecipeDefinition(getIcLoraRecipe(id));
}
