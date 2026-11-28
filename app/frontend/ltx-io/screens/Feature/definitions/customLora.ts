import type { components } from "../../../../generated/backend-openapi";
import {
  CUSTOM_LORA_DEFAULT_STRENGTH,
  CUSTOM_LORA_HOME_FEATURE,
} from "../../../../lib/custom-lora.ts";
import { buildMemoryMeterModel } from "../../../../lib/memory-meter.ts";
import {
  isForcedPassShape,
  meterCell,
  passShapeToSend,
  pricedCell,
  withVisiblePassShape,
  type ForcedPassShape,
  type ImageSize,
} from "../../../../lib/pass-shape.ts";
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
  createPassShapeFields,
  createVideoSettingsOptionFields,
  fastBudgetIssues,
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
import { readStartFrameFromGeneration } from "./loraRecipeStartFrame.ts";
import { negativePromptField, readNegativePrompt } from "./negativePrompt.ts";

type CreateCustomLoraRequest = components["schemas"]["CreateCustomLoraRequest"];
export type CustomLoraItem = components["schemas"]["CustomLoraItem"];

/**
 * Form for one saved custom LoRA: the video controls and a strength slider. The LoRA is the one
 * whose page is open, so the form has no LoRA picker. Each saved LoRA gets its own store key
 * (`customLoraNavId(lora.id)`). The strength is set here, at generate time, like every LoRA. The
 * Start Frame is optional: with an image the run is image to video, without one it is text to
 * video. The create body names the LoRA by id and never carries a path.
 */
export type CustomLoraSchema = {
  prompt: { kind: "textarea" };
  negativePrompt: { kind: "textarea" };
  model: { kind: "options"; value: OfferingId };
  aspectRatio: { kind: "options"; value: VideoFeatureAspectRatio };
  resolution: { kind: "options"; value: VideoGenerationResolution };
  duration: { kind: "options"; value: VideoGenerationDuration };
  fps: { kind: "options"; value: VideoGenerationFps };
  passShape: { kind: "options"; value: ForcedPassShape | null };
  strength: { kind: "slider"; value: number };
  startFrame: { kind: "image-asset"; value: AssetRef | null };
};

export type CustomLoraValues = FeatureValues<CustomLoraSchema>;

export type CustomLoraContext = VideoFeatureContext & {
  /** The LoRA of the open page. `undefined` while the list loads or when it no longer exists. */
  lora: CustomLoraItem | undefined;
  /** The Start Frame's size, for the ratio Auto runs at. Null while it loads or with no frame. */
  startFrameSize?: ImageSize | null;
};

// Same range as the catalog LoRA recipes. The backend accepts up to 4.
const STRENGTH_MIN = 0;
const STRENGTH_MAX = 3;
const STRENGTH_STEP = 0.05;

const baseDefaults: CustomLoraValues = {
  prompt: "",
  negativePrompt: "",
  model: "ltx-2.5-fast",
  aspectRatio: "16:9",
  resolution: "720p",
  duration: 8,
  fps: 24,
  passShape: null,
  strength: CUSTOM_LORA_DEFAULT_STRENGTH,
  startFrame: null,
};

function loraEntry(params: Record<string, unknown>): Record<string, unknown> | null {
  const loras = params.loras;
  if (!Array.isArray(loras) || loras.length === 0) return null;
  const first: unknown = loras[0];
  return first != null && typeof first === "object"
    ? (first as Record<string, unknown>)
    : null;
}

// Desktop keeps the strength on loras[0]. Remote projects it to a top-level field and strips the array.
function readScale(params: Record<string, unknown>): number {
  const direct = params.scale;
  if (typeof direct === "number") return direct;
  const fromEntry = loraEntry(params)?.scale;
  return typeof fromEntry === "number" ? fromEntry : CUSTOM_LORA_DEFAULT_STRENGTH;
}

/**
 * The saved LoRA a stored generation ran, or null. Every custom LoRA run has the same feature id,
 * so this id tells the runs of one LoRA from another. Desktop keeps it on `loras[0]`. Remote
 * projects it to a top-level field.
 */
export function customLoraIdOfGeneration(spec: unknown): string | null {
  const params = readGenerationParams(spec);
  if (!params) return null;
  const id = params.customLoraId ?? loraEntry(params)?.customLoraId;
  return typeof id === "string" && id.length > 0 ? id : null;
}

/** Whether a stored generation ran this saved LoRA. A LoRA page shows only its own runs. */
export function generationRanCustomLora(spec: unknown, customLoraId: string): boolean {
  return customLoraIdOfGeneration(spec) === customLoraId;
}

/** The saved LoRAs of these runs, each once, the LoRA of the newest run first. */
export function recentCustomLoraIds(
  generations: readonly { spec: unknown }[],
): string[] {
  const seen = new Set<string>();
  for (const generation of generations) {
    const id = customLoraIdOfGeneration(generation.spec);
    if (id !== null) seen.add(id);
  }
  return [...seen];
}

function restoreAspectRatio(value: unknown): VideoFeatureAspectRatio {
  if (value === "auto" || isVideoGenerationAspectRatio(value)) return value;
  return baseDefaults.aspectRatio;
}

export function createCustomLoraDefinition(): FeatureDefinition<
  CustomLoraSchema,
  CreateCustomLoraRequest,
  CustomLoraContext,
  typeof CUSTOM_LORA_HOME_FEATURE.id
> {
  function seedValues(context: CustomLoraContext): CustomLoraValues {
    return withDefaultOffering(context.specs, baseDefaults);
  }

  function fromGeneration(
    spec: unknown,
    context: CustomLoraContext,
  ): CustomLoraValues | null {
    const params = readGenerationParams(spec);
    if (!params || typeof params.prompt !== "string") return null;
    return {
      prompt: params.prompt,
      negativePrompt: readNegativePrompt(spec),
      model: restoreOfferingId(params.model, context.specs) ?? baseDefaults.model,
      aspectRatio: restoreAspectRatio(params.aspectRatio),
      resolution: isSupportedVideoResolution(params.resolution)
        ? params.resolution
        : baseDefaults.resolution,
      duration: isSupportedVideoDuration(params.duration)
        ? params.duration
        : baseDefaults.duration,
      fps: isSupportedVideoFps(params.fps) ? params.fps : baseDefaults.fps,
      passShape: isForcedPassShape(params.passShape) ? params.passShape : null,
      strength: readScale(params),
      startFrame: readStartFrameFromGeneration(spec),
    };
  }

  return {
    id: CUSTOM_LORA_HOME_FEATURE.id,
    title: CUSTOM_LORA_HOME_FEATURE.title,
    defaults: baseDefaults,
    initialValues: seedValues,
    resetValues: seedValues,
    form: (values, context): FeatureFormModel<CustomLoraSchema> => {
      // `auto` sizes the video from the Start Frame, so it needs one.
      const [model, ...rest] = createVideoSettingsOptionFields<CustomLoraSchema>(
        values,
        context.specs,
        {
          includeAuto: values.startFrame != null,
          memoryMeter: true,
          startFrameSize: context.startFrameSize,
        },
      );
      return {
        fields: [
          {
            kind: "image-asset",
            id: "startFrame",
            label: "Start Frame (optional)",
            dataKey: "startFrame",
          },
          {
            kind: "textarea",
            id: "prompt",
            label: "Prompt",
            dataKey: "prompt",
            placeholder: "Describe the scene. Add the trigger words your LoRA needs.",
          },
          negativePromptField<CustomLoraSchema>(),
          model,
          ...rest,
          ...createPassShapeFields(
            pricedCell(context.specs, values, context.startFrameSize),
            context.specs,
          ),
          {
            kind: "slider",
            id: "strength",
            label: "LoRA strength",
            dataKey: "strength",
            min: STRENGTH_MIN,
            max: STRENGTH_MAX,
            step: STRENGTH_STEP,
          },
        ],
        memory: buildMemoryMeterModel(
          context.specs,
          meterCell(context.specs, values, context.startFrameSize),
          { fixFields: ["passShape", "duration", "resolution"] },
        ),
      };
    },
    applyChange: (
      values,
      change: FeatureFieldChange<CustomLoraSchema>,
      context,
    ): CustomLoraValues => {
      switch (change.dataKey) {
        case "strength":
          return { ...values, strength: change.value };
        default:
          // clampVideoFields (via applyVideoSettingsChange) keeps the other fields as they are.
          return applyVideoSettingsChange<CustomLoraSchema>(
            values,
            change,
            context.specs,
            withDefaultOffering(context.specs, baseDefaults),
          );
      }
    },
    normalize: (values, context) => {
      // Without a start frame there is nothing to size from, so `auto` has no meaning.
      const aspectRatio =
        values.startFrame == null && values.aspectRatio === "auto"
          ? baseDefaults.aspectRatio
          : values.aspectRatio;
      const defaults = withDefaultOffering(context.specs, baseDefaults);
      const clamped = clampVideoFields({ ...values, aspectRatio }, context.specs, defaults);
      return withVisiblePassShape(
        context.specs,
        clamped,
        pricedCell(context.specs, clamped, context.startFrameSize),
      );
    },
    fromGeneration,
    validate: (values, context): ValidationIssue[] => {
      const issues: ValidationIssue[] = [];
      const lora = context.lora;
      if (lora === undefined) {
        issues.push({
          id: "custom-lora-required",
          fieldId: "customLora",
          message: "This LoRA was not found.",
        });
      }
      if (values.prompt.trim().length === 0) {
        issues.push({
          id: "prompt-required",
          fieldId: "prompt",
          message: "Prompt is required.",
        });
      }
      issues.push(
        ...fastBudgetIssues(
          context.specs,
          meterCell(context.specs, values, context.startFrameSize),
        ),
      );
      return issues;
    },
    toCreateBody: (values, context): CreateCustomLoraRequest => {
      const lora = context.lora;
      if (lora === undefined) {
        throw new Error("This LoRA was not found.");
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
          negativePrompt: values.negativePrompt.trim(),
          customLoraId: lora.id,
          scale: values.strength,
          promptProvenance: "typed",
          passShape: passShapeToSend(context.specs, values, context.startFrameSize),
        },
        ...(values.startFrame != null
          ? { inputs: { startFrame: { assetId: values.startFrame.assetId } } }
          : {}),
      };
    },
    isReady: (values, context) =>
      context.lora?.downloaded === true &&
      isLocalVideoSelectionReady(values, context.specs),
    isUnavailable: (values, context) =>
      !localVideoHasCompatibleOptions(values, context.specs),
  };
}
