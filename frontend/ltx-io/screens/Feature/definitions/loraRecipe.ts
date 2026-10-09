import type { components } from "../../../../generated/backend-openapi";
import type {
  LoraRecipeDefinition as LoraRecipeMeta,
  LoraRecipeId,
} from "../../../../lib/lora-recipes.ts";
import { recipeHasEndFrame } from "../../../../lib/lora-recipes.ts";
import {
  restoreOfferingId,
  withDefaultOffering,
  type OfferingId,
  isVideoGenerationAspectRatio,
  type VideoGenerationDuration,
  type VideoGenerationFps,
  type VideoGenerationResolution,
} from "../../../../lib/video-generation-model-specs.ts";
import { readGenerationParams } from "../../../lib/resultsFeedModel.ts";
import {
  clampVideoFields,
  isLocalVideoSelectionReady,
  localVideoHasCompatibleOptions,
  type VideoFeatureAspectRatio,
  type VideoFeatureContext,
} from "../../../lib/videoFieldPolicy.ts";
import {
  applyVideoSettingsChange,
  createVideoSettingsOptionFields,
  isSupportedVideoDuration,
  isSupportedVideoFps,
  isSupportedVideoResolution,
} from "../../../lib/videoSettingsControls.ts";
import {
  type AssetRef,
  type FeatureDefinition,
  type FeatureFieldChange,
  type FeatureFormModel,
  type FeatureValues,
  type ValidationIssue,
} from "../types.ts";
import {
  resolveSupportedModel,
  resolveVariantChoice,
  unsupportedModelReason,
  variantChoiceId,
  variantChoiceOptions,
  variantChoiceWarning,
  type CatalogSupportedModels,
  type CatalogVariantChoice,
} from "../../../lib/catalogVariantChoice.ts";
import { readEndFrameFromGeneration, readStartFrameFromGeneration } from "./loraRecipeStartFrame.ts";

type CreateLoraRecipeRequest = components["schemas"]["CreateLoraRecipeRequest"];

/**
 * Shared schema for every listed LoRA recipe: video controls plus a strength
 * slider. i2v recipes also carry a Start Frame; Transition also carries End
 * Frame. t2v recipes leave both null and omit the fields from the form. The
 * base t2v/i2v pages stay `loras: []`; this is the only form that carries a
 * LoRA scale, and it never carries a filesystem
 * ref.
 */
export type LoraRecipeSchema = {
  prompt: { kind: "textarea" };
  model: { kind: "options"; value: OfferingId };
  variant: { kind: "options"; value: string };
  aspectRatio: { kind: "options"; value: VideoFeatureAspectRatio };
  resolution: { kind: "options"; value: VideoGenerationResolution };
  duration: { kind: "options"; value: VideoGenerationDuration };
  fps: { kind: "options"; value: VideoGenerationFps };
  strength: { kind: "slider"; value: number };
  startFrame: { kind: "image-asset"; value: AssetRef | null };
  endFrame: { kind: "image-asset"; value: AssetRef | null };
};

export type LoraRecipeValues = FeatureValues<LoraRecipeSchema>;

export type LoraRecipeContext = VideoFeatureContext & {
  seed?: AssetRef | null;
  endSeed?: AssetRef | null;
  catalogVariants?: CatalogVariantChoice[];
  /** The catalog `supported_models`. A model outside it is listed but disabled. */
  supportedModels?: CatalogSupportedModels;
};

// UI strength range. Intentionally tighter than the backend LoraEntry scale
// bound (0..4): weights above ~3 are unusable in practice, so the slider caps at
// 3 to keep the useful region easy to hit. The default comes from the recipe registry.
const STRENGTH_MIN = 0;
const STRENGTH_MAX = 3;
const STRENGTH_STEP = 0.05;

function isI2vRecipe(recipe: LoraRecipeMeta): boolean {
  return recipe.mode === "i2v";
}

function baseDefaults(recipe: LoraRecipeMeta): LoraRecipeValues {
  return {
    prompt: recipe.seedPrompt,
    model: "ltx-2.5-fast",
    variant: "",
    aspectRatio: isI2vRecipe(recipe) ? "auto" : "16:9",
    resolution: "720p",
    duration: 8,
    fps: 24,
    strength: recipe.defaultStrength,
    startFrame: null,
    endFrame: null,
  };
}

function readStoredVariantId(params: Record<string, unknown>): string {
  const loras = params.loras;
  if (!Array.isArray(loras) || loras.length === 0) return "";
  const variantId = (loras[0] as { variantId?: unknown }).variantId;
  return typeof variantId === "string" ? variantId : "";
}

function readScale(params: Record<string, unknown>, fallback: number): number {
  // Remote projects strength to a top-level `scale` (the loras array with the
  // filesystem ref is stripped from RemoteGeneration); desktop keeps it on
  // loras[0].scale.
  const direct = params.scale;
  if (typeof direct === "number") return direct;
  const loras = params.loras;
  if (Array.isArray(loras) && loras.length > 0) {
    const scale = (loras[0] as { scale?: unknown }).scale;
    if (typeof scale === "number") return scale;
  }
  return fallback;
}

function restoreAspectRatio(
  value: unknown,
  recipe: LoraRecipeMeta,
  fallback: VideoFeatureAspectRatio,
): VideoFeatureAspectRatio {
  if (value === "auto") return isI2vRecipe(recipe) ? "auto" : fallback;
  if (isVideoGenerationAspectRatio(value)) return value;
  return fallback;
}

function modelOfferingIds(
  values: LoraRecipeValues,
  context: LoraRecipeContext,
  includeAuto: boolean,
): string[] {
  const [model] = createVideoSettingsOptionFields<LoraRecipeSchema>(
    values,
    context.specs,
    { includeAuto },
  );
  return model.options.map((option) => String(option.value));
}

function variantParam(
  variants: readonly CatalogVariantChoice[] | undefined,
  selected: string,
  offeringId: string,
  offeringIds: readonly string[],
): { variantId: string } | Record<string, never> {
  const variantId = variantChoiceId(
    resolveVariantChoice(variants ?? [], selected, offeringId, offeringIds),
  );
  return variantId ? { variantId } : {};
}

function recipeSettingFields(
  values: LoraRecipeValues,
  context: LoraRecipeContext,
  includeAuto: boolean,
) {
  const [model, ...rest] = createVideoSettingsOptionFields<LoraRecipeSchema>(
    values,
    context.specs,
    { includeAuto },
  );
  const offeringIds = model.options.map((option) => String(option.value));
  const choice = resolveVariantChoice(
    context.catalogVariants ?? [],
    values.variant,
    values.model,
    offeringIds,
  );
  const variantOptions = variantChoiceOptions(choice, values.model, "LoRA");
  return [
    {
      ...model,
      options: model.options.map((option) => {
        const disabledReason = unsupportedModelReason(
          context.supportedModels,
          String(option.value),
          "LoRA",
        );
        if (disabledReason != null) return { ...option, disabledReason };
        const warning = variantChoiceWarning(choice, String(option.value), "LoRA");
        return warning == null ? option : { ...option, warning };
      }),
    },
    ...(variantOptions
      ? [
          {
            kind: "options" as const,
            id: "variant",
            label: "Variant",
            dataKey: "variant" as const,
            forcePicker: true,
            options: variantOptions,
          },
        ]
      : []),
    ...rest,
  ];
}

export function createLoraRecipeDefinition(
  recipe: LoraRecipeMeta,
): FeatureDefinition<
  LoraRecipeSchema,
  CreateLoraRecipeRequest,
  LoraRecipeContext,
  LoraRecipeId
> {
  const defaults = baseDefaults(recipe);
  const i2v = isI2vRecipe(recipe);
  const needsEndFrame = recipeHasEndFrame(recipe);

  function seedValues(context: LoraRecipeContext): LoraRecipeValues {
    return {
      ...withDefaultOffering(context.specs, defaults),
      startFrame: i2v ? (context.seed ?? null) : null,
      endFrame: needsEndFrame ? (context.endSeed ?? null) : null,
    };
  }

  function fromGeneration(
    spec: unknown,
    context: LoraRecipeContext,
  ): LoraRecipeValues | null {
    const params = readGenerationParams(spec);
    if (!params || typeof params.prompt !== "string") return null;
    return {
      prompt: params.prompt,
      model: restoreOfferingId(params.model, context.specs) ?? defaults.model,
      variant: readStoredVariantId(params),
      aspectRatio: restoreAspectRatio(params.aspectRatio, recipe, defaults.aspectRatio),
      resolution: isSupportedVideoResolution(params.resolution)
        ? params.resolution
        : defaults.resolution,
      duration: isSupportedVideoDuration(params.duration)
        ? params.duration
        : defaults.duration,
      fps: isSupportedVideoFps(params.fps) ? params.fps : defaults.fps,
      strength: readScale(params, recipe.defaultStrength),
      startFrame: i2v ? (readStartFrameFromGeneration(spec) ?? context.seed ?? null) : null,
      endFrame: needsEndFrame
        ? (readEndFrameFromGeneration(spec) ?? context.endSeed ?? null)
        : null,
    };
  }

  return {
    id: recipe.id,
    title: recipe.title,
    defaults,
    initialValues: seedValues,
    resetValues: seedValues,
    form: (values, context): FeatureFormModel<LoraRecipeSchema> => ({
      fields: [
        ...(i2v
          ? [
              {
                kind: "image-asset" as const,
                id: "startFrame",
                label: "Start Frame",
                dataKey: "startFrame" as const,
              },
              ...(needsEndFrame
                ? [
                    {
                      kind: "image-asset" as const,
                      id: "endFrame",
                      label: "End Frame",
                      dataKey: "endFrame" as const,
                    },
                  ]
                : []),
            ]
          : []),
        {
          kind: "textarea",
          id: "prompt",
          label: "Prompt",
          dataKey: "prompt",
          placeholder: recipe.placeholder,
        },
        ...recipeSettingFields(values, context, i2v),
        {
          kind: "slider",
          id: "strength",
          label: "Style strength",
          dataKey: "strength",
          min: STRENGTH_MIN,
          max: STRENGTH_MAX,
          step: STRENGTH_STEP,
        },
      ],
    }),
    applyChange: (
      values,
      change: FeatureFieldChange<LoraRecipeSchema>,
      context,
    ): LoraRecipeValues => {
      if (change.dataKey === "strength") {
        return { ...values, strength: change.value };
      }
      if (change.dataKey === "variant") {
        return { ...values, variant: change.value };
      }
      // clampVideoFields (via applyVideoSettingsChange) preserves prompt +
      // strength + startFrame + endFrame.
      return applyVideoSettingsChange<LoraRecipeSchema>(
        values,
        change,
        context.specs,
        withDefaultOffering(context.specs, defaults),
      );
    },
    normalize: (values, context) => {
      const base = clampVideoFields(
        values,
        context.specs,
        withDefaultOffering(context.specs, defaults),
      );
      const model = resolveSupportedModel(
        base.model,
        modelOfferingIds(base, context, i2v) as (typeof base.model)[],
        context.supportedModels,
      );
      const clamped =
        model === base.model
          ? base
          : clampVideoFields(
              { ...base, model },
              context.specs,
              withDefaultOffering(context.specs, defaults),
            );
      const variant = variantChoiceId(
        resolveVariantChoice(
          context.catalogVariants ?? [],
          clamped.variant ?? "",
          clamped.model,
          modelOfferingIds(clamped, context, i2v),
        ),
      );
      return variant === clamped.variant ? clamped : { ...clamped, variant };
    },
    fromGeneration,
    validate: (values, context): ValidationIssue[] => {
      const issues: ValidationIssue[] = [];
      const unsupported = unsupportedModelReason(
        context.supportedModels,
        values.model,
        "LoRA",
      );
      if (unsupported != null) {
        issues.push({
          id: "model-unsupported",
          fieldId: "model",
          alwaysRevealed: true,
          message: unsupported,
        });
      }
      if (values.prompt.trim().length === 0) {
        issues.push({
          id: "prompt-required",
          fieldId: "prompt",
          message: "Prompt is required.",
        });
      }
      if (i2v && values.startFrame == null) {
        issues.push({
          id: "start-frame-required",
          fieldId: "startFrame",
          message: "Start Frame is required.",
        });
      }
      if (needsEndFrame && values.endFrame == null) {
        issues.push({
          id: "end-frame-required",
          fieldId: "endFrame",
          message: "End Frame is required.",
        });
      }
      return issues;
    },
    toCreateBody: (values, context): CreateLoraRecipeRequest => {
      if (i2v && values.startFrame == null) {
        throw new Error("Start Frame is required.");
      }
      if (needsEndFrame && values.endFrame == null) {
        throw new Error("End Frame is required.");
      }
      return {
        contract_version: 1,
        params: {
          prompt: values.prompt.trim(),
          model: values.model,
          aspectRatio: values.aspectRatio,
          resolution: values.resolution,
          duration: values.duration,
          fps: values.fps,
          cameraMotion: "none",
          negativePrompt: "",
          // catalogId on the wire; the backend resolves the installed path. No ref.
          catalogId: recipe.catalogId,
          scale: values.strength,
          ...variantParam(
            context.catalogVariants,
            values.variant,
            values.model,
            modelOfferingIds(values, context, i2v),
          ),
          promptProvenance: "typed",
        },
        ...(i2v && values.startFrame != null
          ? {
              inputs: {
                startFrame: { assetId: values.startFrame.assetId },
                ...(needsEndFrame && values.endFrame != null
                  ? { endFrame: { assetId: values.endFrame.assetId } }
                  : {}),
              },
            }
          : {}),
      };
    },
    isReady: (values, context) =>
      isLocalVideoSelectionReady(values, context.specs),
    isUnavailable: (values, context) =>
      !localVideoHasCompatibleOptions(values, context.specs),
  };
}
