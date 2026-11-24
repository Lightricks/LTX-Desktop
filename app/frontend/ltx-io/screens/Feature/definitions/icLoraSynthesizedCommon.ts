import {
  type CatalogSupportedModels,
  type CatalogVariantChoice,
} from "../../../lib/catalogVariantChoice.ts";
import {
  type OfferingId,
  withDefaultOffering,
  type VideoGenerationDuration,
  type VideoGenerationFps,
  type VideoGenerationResolution,
} from "../../../../lib/video-generation-model-specs.ts";
import {
  clampVideoFields,
  localVideoHasCompatibleOptions,
  type ClampedVideoFields,
  type VideoFeatureAspectRatio,
  type VideoFeatureContext,
} from "../../../lib/videoFieldPolicy.ts";
import {
  createVideoSettingsOptionFields,
  isSupportedVideoDuration,
  isSupportedVideoFps,
  isSupportedVideoResolution,
} from "../../../lib/videoSettingsControls.ts";
import type {
  FeatureFieldSchema,
  FeatureValues,
  FieldOption,
} from "../types.ts";
import { isIcLoraSupportedFps } from "./icLoraFps.ts";
import { normalizeIcLoraModelAndVariant } from "./icLoraNormalize.ts";
import { readStoredLora } from "./icLoraRecipe.ts";
import {
  IC_LORA_AUDIO_OPTIONS,
  icLoraModelFields,
  STRENGTH_MAX,
  STRENGTH_MIN,
  STRENGTH_STEP,
  type IcLoraRecipeAudioMode,
} from "./icLoraRecipeForm.ts";
import { IC_LORA_RESOLUTIONS } from "./icLoraResolution.ts";
import { readNegativePrompt } from "./negativePrompt.ts";
import { hasAdvertisedCapability, resolveCapableModel } from "./videoFeature.ts";

/** A recipe that makes its own control clip has no source clip, so it has no Source audio. */
export type IcLoraAudioMode = Exclude<IcLoraRecipeAudioMode, "source">;

export type SynthesizedIcLoraSchemaBase = FeatureFieldSchema & {
  prompt: { kind: "textarea" };
  negativePrompt: { kind: "textarea" };
  model: { kind: "options"; value: OfferingId };
  variant: { kind: "options"; value: string };
  aspectRatio: { kind: "options"; value: VideoFeatureAspectRatio };
  resolution: { kind: "options"; value: VideoGenerationResolution };
  duration: { kind: "options"; value: VideoGenerationDuration };
  fps: { kind: "options"; value: VideoGenerationFps };
  audioMode: { kind: "options"; value: IcLoraAudioMode };
  loraStrength: { kind: "slider"; value: number };
};

export type SynthesizedIcLoraContext = VideoFeatureContext & {
  catalogVariants?: CatalogVariantChoice[];
  /** The catalog `supported_models`. A model outside it is listed but disabled. */
  supportedModels?: CatalogSupportedModels;
};

export type SynthesizedIcLoraCatalogSettings = {
  audio_mode?: string | null;
  default_negative_prompt?: string | null;
};

export const AUDIO_OPTIONS = IC_LORA_AUDIO_OPTIONS.filter(
  (option): option is FieldOption<IcLoraAudioMode> => option.value !== "source",
);

export function isIcLoraAudioMode(value: unknown): value is IcLoraAudioMode {
  return AUDIO_OPTIONS.some((option) => option.value === value);
}

const SYNTHESIZED_RESOLUTIONS: readonly string[] = IC_LORA_RESOLUTIONS;

export function isSynthesizedIcLoraResolution(
  value: unknown,
): value is (typeof IC_LORA_RESOLUTIONS)[number] {
  return typeof value === "string" && SYNTHESIZED_RESOLUTIONS.includes(value);
}

export function readSynthesizedCommonParams(
  params: Record<string, unknown>,
  context: SynthesizedIcLoraContext,
  defaults: {
    model: OfferingId;
    resolution: VideoGenerationResolution;
    duration: VideoGenerationDuration;
    fps: VideoGenerationFps;
    audioMode: IcLoraAudioMode;
    loraStrength: number;
  },
) {
  const stored = readStoredLora(params, defaults.loraStrength);
  return {
    prompt: typeof params.prompt === "string" ? params.prompt : "",
    negativePrompt: readNegativePrompt({ params }),
    model: resolveCapableModel(params.model, context.specs, "ic_lora", defaults.model),
    variant: stored.variantId,
    resolution: isSupportedVideoResolution(params.resolution)
      ? params.resolution
      : defaults.resolution,
    duration: isSupportedVideoDuration(params.duration) ? params.duration : defaults.duration,
    fps: isSupportedVideoFps(params.fps) ? params.fps : defaults.fps,
    audioMode: isIcLoraAudioMode(params.audioMode) ? params.audioMode : defaults.audioMode,
    loraStrength: stored.scale,
  };
}

export function normalizeSynthesizedIcLoraValues<TSchema extends SynthesizedIcLoraSchemaBase>(
  values: FeatureValues<TSchema>,
  context: SynthesizedIcLoraContext,
  defaults: FeatureValues<TSchema>,
  effectiveAspectRatio?: VideoFeatureAspectRatio,
): FeatureValues<TSchema> {
  // A run saved with 1440p or 4K is not offered on synthesized recipes. It reopens at 1080p.
  const local = isSynthesizedIcLoraResolution(values.resolution)
    ? values
    : { ...values, resolution: "1080p" as const };
  const priced =
    effectiveAspectRatio !== undefined
      ? { ...local, aspectRatio: effectiveAspectRatio }
      : local;
  const clampedPriced = clampVideoFields(
    priced,
    context.specs,
    withDefaultOffering(context.specs, defaults),
  );
  const clamped: FeatureValues<TSchema> =
    effectiveAspectRatio !== undefined
      ? { ...clampedPriced, aspectRatio: values.aspectRatio }
      : clampedPriced;
  const { model, variant } = normalizeIcLoraModelAndVariant(clamped, context, defaults.model);
  return model === clamped.model && variant === clamped.variant
    ? clamped
    : { ...clamped, model, variant };
}

export function synthesizedIcLoraVideoOptionFields<TSchema extends SynthesizedIcLoraSchemaBase>(
  values: FeatureValues<TSchema>,
  context: SynthesizedIcLoraContext,
  effectiveAspectRatio?: VideoFeatureAspectRatio,
) {
  const priced =
    effectiveAspectRatio !== undefined
      ? { ...values, aspectRatio: effectiveAspectRatio }
      : values;
  const [, aspectRatio, resolution, duration, fps] =
    createVideoSettingsOptionFields<TSchema>(priced, context.specs);
  // The recipe runs the local cells only: 270p to 1080p. It has no tiled high-res run.
  const filteredResolution = {
    ...resolution,
    options: resolution.options.filter((option) => isSynthesizedIcLoraResolution(option.value)),
  };
  // The recipe runs at a rate of the shared IC-LoRA list, not any rate the model offers.
  const filteredFps = {
    ...fps,
    options: fps.options.filter((option) => isIcLoraSupportedFps(option.value)),
  };
  return {
    aspectRatio,
    resolution: filteredResolution,
    duration,
    fps: filteredFps,
    tail: [
      filteredResolution,
      duration,
      filteredFps,
      {
        kind: "options" as const,
        id: "audioMode",
        label: "Audio",
        dataKey: "audioMode" as const,
        options: AUDIO_OPTIONS,
      },
      ...icLoraModelFields(values, context),
      {
        kind: "slider" as const,
        id: "loraStrength",
        label: "Style strength",
        dataKey: "loraStrength" as const,
        min: STRENGTH_MIN,
        max: STRENGTH_MAX,
        step: STRENGTH_STEP,
      },
    ],
  };
}

export function isSynthesizedIcLoraUnavailable(
  values: ClampedVideoFields,
  context: SynthesizedIcLoraContext,
): boolean {
  return (
    !hasAdvertisedCapability(context.specs, "ic_lora") ||
    !localVideoHasCompatibleOptions(values, context.specs)
  );
}
