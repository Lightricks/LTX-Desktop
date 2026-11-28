import type { components } from "../../../../generated/backend-openapi";
import { clampKeyframeStrength } from "../../../../lib/keyframe-strength.ts";
import {
  lastFrameFromDuration,
  retimeKeyframesForSettings,
} from "../../../../lib/keyframe-timeline.ts";
import { buildMemoryMeterModel } from "../../../../lib/memory-meter.ts";
import { LOCAL_MULTI_KEYFRAME_MAX_COUNT } from "../../../../lib/multi-keyframe.ts";
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
  getOfferingVideoGenerationModelSpecs,
  restoreOfferingId,
  withDefaultOffering,
  type OfferingId,
  isVideoGenerationAspectRatio,
  type VideoGenerationDuration,
  type VideoGenerationFps,
  type VideoGenerationModelSpecsResponse,
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
  type FeatureDefinition,
  type FeatureFieldChange,
  type FeatureFormModel,
  type FeatureKeyframe,
  type FeatureValues,
  type ValidationIssue,
} from "../types.ts";
import {
  KEYFRAMES_EXAMPLE_NEGATIVE_PROMPT,
  KEYFRAMES_EXAMPLE_PROMPT,
  KEYFRAMES_EXAMPLE_SETTINGS,
  exampleKeyframes,
  type KeyframesExampleSeed,
} from "./keyframesExample.ts";
import { negativePromptField, readNegativePrompt } from "./negativePrompt.ts";
import { readAssetRef } from "./videoFeature.ts";

type CreateKeyframesRequest = components["schemas"]["CreateKeyframesRequest"];

export type KeyframesSchema = {
  prompt: { kind: "textarea" };
  negativePrompt: { kind: "textarea" };
  model: { kind: "options"; value: OfferingId };
  aspectRatio: { kind: "options"; value: VideoFeatureAspectRatio };
  resolution: { kind: "options"; value: VideoGenerationResolution };
  duration: { kind: "options"; value: VideoGenerationDuration };
  fps: { kind: "options"; value: VideoGenerationFps };
  passShape: { kind: "options"; value: ForcedPassShape | null };
  keyframes: { kind: "keyframes"; value: FeatureKeyframe[] };
};

export type KeyframesValues = FeatureValues<KeyframesSchema>;

export type KeyframesContext = VideoFeatureContext & {
  /** The earliest keyframe's size, for the ratio Auto runs at. Null while it loads. */
  startFrameSize?: ImageSize | null;
  /** The packaged example stills for a first open. Null when the form must not start from them. */
  seed?: KeyframesExampleSeed | null;
};

export const KEYFRAMES_DEFAULTS: KeyframesValues = {
  prompt: "",
  negativePrompt: "",
  model: "ltx-2.5-fast",
  aspectRatio: "auto",
  resolution: "540p",
  duration: 8,
  fps: 24,
  passShape: null,
  keyframes: [],
};

/** How many stills the chosen model takes. Missing means the local limit. */
export function keyframesMaxCount(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  model: OfferingId,
): number {
  const row = getOfferingVideoGenerationModelSpecs(specs).find((item) => item.model === model);
  return row?.spec.capabilities?.multi_keyframe_max_count ?? LOCAL_MULTI_KEYFRAME_MAX_COUNT;
}

/** Whether a downloaded offering can run keyframes. */
export function supportsKeyframes(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
): boolean {
  return getOfferingVideoGenerationModelSpecs(specs).some(
    (item) => item.spec.capabilities?.multi_keyframe === true,
  );
}

/** The earliest still. It sets the ratio that Auto runs at. */
export function earliestKeyframe(
  keyframes: readonly FeatureKeyframe[],
): FeatureKeyframe | undefined {
  return keyframes.reduce<FeatureKeyframe | undefined>(
    (first, keyframe) =>
      first === undefined || keyframe.frameIndex < first.frameIndex ? keyframe : first,
    undefined,
  );
}

function readGenerationKeyframes(spec: unknown): FeatureKeyframe[] | null {
  if (!spec || typeof spec !== "object") return null;
  const inputs = (spec as { inputs?: unknown }).inputs;
  if (!inputs || typeof inputs !== "object" || Array.isArray(inputs)) return null;
  const raw = (inputs as { keyframes?: unknown }).keyframes;
  if (!Array.isArray(raw) || raw.length === 0) return null;
  const keyframes: FeatureKeyframe[] = [];
  for (const [index, item] of raw.entries()) {
    if (!item || typeof item !== "object") return null;
    const asset = readAssetRef(item);
    const { frameIndex, strength } = item as { frameIndex?: unknown; strength?: unknown };
    if (!asset || typeof frameIndex !== "number" || !Number.isInteger(frameIndex)) return null;
    keyframes.push({
      id: `keyframe-${index}`,
      asset,
      frameIndex,
      strength: clampKeyframeStrength(strength),
    });
  }
  return keyframes;
}

export function fromGeneration(
  spec: unknown,
  context?: VideoFeatureContext,
): KeyframesValues | null {
  const params = readGenerationParams(spec);
  const keyframes = readGenerationKeyframes(spec);
  if (!params || !keyframes) return null;

  return {
    prompt: typeof params.prompt === "string" ? params.prompt : "",
    negativePrompt: readNegativePrompt(spec),
    model: restoreOfferingId(params.model, context?.specs) ?? KEYFRAMES_DEFAULTS.model,
    aspectRatio:
      params.aspectRatio === "auto" || isVideoGenerationAspectRatio(params.aspectRatio)
        ? params.aspectRatio
        : KEYFRAMES_DEFAULTS.aspectRatio,
    resolution: isSupportedVideoResolution(params.resolution)
      ? params.resolution
      : KEYFRAMES_DEFAULTS.resolution,
    duration: isSupportedVideoDuration(params.duration)
      ? params.duration
      : KEYFRAMES_DEFAULTS.duration,
    fps: isSupportedVideoFps(params.fps) ? params.fps : KEYFRAMES_DEFAULTS.fps,
    passShape: isForcedPassShape(params.passShape) ? params.passShape : null,
    keyframes,
  };
}

export function validateKeyframes(
  values: KeyframesValues,
  context?: KeyframesContext,
): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  if (values.keyframes.length === 0) {
    issues.push({
      id: "keyframes-required",
      fieldId: "keyframes",
      message: "Add at least one keyframe.",
    });
  }
  if (values.prompt.trim() === "") {
    issues.push({
      id: "prompt-required",
      fieldId: "prompt",
      message: "Prompt is required.",
    });
  }
  const lastFrame = lastFrameFromDuration(values.duration, values.fps);
  const frames = values.keyframes.map((keyframe) => keyframe.frameIndex);
  if (
    new Set(frames).size !== frames.length ||
    frames.some((frame) => frame < 0 || frame > lastFrame)
  ) {
    issues.push({
      id: "keyframes-frames-invalid",
      fieldId: "keyframes",
      message: "Each keyframe needs its own frame inside the video.",
      alwaysRevealed: true,
    });
  }
  issues.push(
    ...fastBudgetIssues(
      context?.specs,
      meterCell(context?.specs, values, context?.startFrameSize),
    ),
  );
  return issues;
}

export function toCreateBody(
  values: KeyframesValues,
  context: KeyframesContext,
): CreateKeyframesRequest {
  return {
    contract_version: 1,
    inputs: {
      // The local `id` is a React key. The create body never carries it.
      keyframes: [...values.keyframes]
        .sort((a, b) => a.frameIndex - b.frameIndex)
        .map((keyframe) => ({
          assetId: keyframe.asset.assetId,
          frameIndex: keyframe.frameIndex,
          strength: clampKeyframeStrength(keyframe.strength),
        })),
    },
    params: {
      prompt: values.prompt.trim(),
      model: values.model,
      aspectRatio: values.aspectRatio,
      resolution: values.resolution,
      duration: values.duration,
      fps: values.fps,
      cameraMotion: "none",
      negativePrompt: values.negativePrompt.trim(),
      promptProvenance: "typed",
      passShape: passShapeToSend(context.specs, values, context.startFrameSize),
    },
  };
}

export function buildKeyframesForm(
  values: KeyframesValues,
  context: KeyframesContext,
): FeatureFormModel<KeyframesSchema> {
  return {
    fields: [
      {
        kind: "keyframes",
        id: "keyframes",
        label: "Keyframes",
        dataKey: "keyframes",
        maxCount: keyframesMaxCount(context.specs, values.model),
        durationKey: "duration",
        fpsKey: "fps",
      },
      {
        kind: "textarea",
        id: "prompt",
        label: "Prompt",
        dataKey: "prompt",
        placeholder: "Describe how the video moves between the keyframes...",
      },
      negativePromptField<KeyframesSchema>(),
      ...createVideoSettingsOptionFields<KeyframesSchema>(values, context.specs, {
        includeAuto: true,
        memoryMeter: true,
        startFrameSize: context.startFrameSize,
      }),
      ...createPassShapeFields(
        pricedCell(context.specs, values, context.startFrameSize),
        context.specs,
      ),
    ],
    memory: buildMemoryMeterModel(
      context.specs,
      meterCell(context.specs, values, context.startFrameSize),
      { fixFields: ["passShape", "duration", "resolution"] },
    ),
  };
}

/**
 * A new length or frame rate moves the timeline, so the stills move with it. A clamp can also
 * change them, for example when the model does not offer the chosen length.
 */
function retimeKeyframes(before: KeyframesValues, after: KeyframesValues): KeyframesValues {
  if (before.duration === after.duration && before.fps === after.fps) return after;
  return {
    ...after,
    keyframes: retimeKeyframesForSettings(
      after.keyframes,
      { duration: before.duration, fps: before.fps },
      { duration: after.duration, fps: after.fps },
    ),
  };
}

function withDefaults(context: KeyframesContext): KeyframesValues {
  return withDefaultOffering(context.specs, KEYFRAMES_DEFAULTS);
}

/** A first open starts from the packaged example. Reset goes back to an empty form. */
export function seedKeyframesValues(context: KeyframesContext): KeyframesValues {
  const values = withDefaults(context);
  if (!context.seed) return values;
  return {
    ...values,
    ...KEYFRAMES_EXAMPLE_SETTINGS,
    prompt: KEYFRAMES_EXAMPLE_PROMPT,
    negativePrompt: KEYFRAMES_EXAMPLE_NEGATIVE_PROMPT,
    keyframes: exampleKeyframes(context.seed),
  };
}

export function applyKeyframesChange(
  values: KeyframesValues,
  change: FeatureFieldChange<KeyframesSchema>,
  context: KeyframesContext,
): KeyframesValues {
  return retimeKeyframes(
    values,
    applyVideoSettingsChange(values, change, context.specs, withDefaults(context)),
  );
}

export const keyframesDefinition: FeatureDefinition<
  KeyframesSchema,
  CreateKeyframesRequest,
  KeyframesContext,
  "keyframes"
> = {
  id: "keyframes",
  title: "Keyframes",
  defaults: KEYFRAMES_DEFAULTS,
  initialValues: seedKeyframesValues,
  resetValues: withDefaults,
  form: buildKeyframesForm,
  applyChange: applyKeyframesChange,
  normalize: (values, context) => {
    const clamped = retimeKeyframes(
      values,
      clampVideoFields(values, context.specs, withDefaults(context)),
    );
    return withVisiblePassShape(
      context.specs,
      clamped,
      pricedCell(context.specs, clamped, context.startFrameSize),
    );
  },
  fromGeneration,
  validate: validateKeyframes,
  toCreateBody,
  isReady: (values, context) => isLocalVideoSelectionReady(values, context.specs),
  isUnavailable: (values, context) =>
    !supportsKeyframes(context.specs) ||
    !localVideoHasCompatibleOptions(values, context.specs),
};
