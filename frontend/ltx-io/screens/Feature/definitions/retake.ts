import type { components } from "../../../../generated/backend-openapi";
import {
  restoreOfferingId,
  withDefaultOffering,
  type OfferingId,
} from "../../../../lib/video-generation-model-specs.ts";
import { roundDurationSeconds } from "../../../lib/roundDuration.ts";
import { readGenerationParams } from "../../../lib/resultsFeedModel.ts";
import {
  applyFeatureFieldChange,
  type AssetRef,
  type FeatureDefinition,
  type FeatureFieldChange,
  type FeatureFormModel,
  type FeatureValues,
  type FieldOption,
  type ValidationIssue,
} from "../types.ts";
import {
  capabilityModelOptions,
  editResolutionFieldOptions,
  editResolutionSize,
  editResolutionKeyFromParams,
  type EditResolutionKey,
  hasAdvertisedCapability,
  normalizeEditResolution,
  readPrompt,
  readVideoGenerationInput,
  resolveCapableModel,
  type VideoInputContext,
  videoLengthUnknownIssue,
  videoRequiredIssue,
} from "./videoFeature.ts";

type CreateRetakeRequest = components["schemas"]["CreateRetakeRequest"];

/** Create always sends replace_audio_and_video until a mode picker exists. */
const RETAKE_CREATE_MODE = "replace_audio_and_video" as const;

export type RetakeSchema = {
  video: { kind: "video-asset"; value: AssetRef | null };
  prompt: { kind: "textarea" };
  model: { kind: "options"; value: OfferingId };
  resolution: { kind: "options"; value: EditResolutionKey };
  startTime: { kind: "slider"; value: number };
  duration: { kind: "slider"; value: number };
};

export type RetakeValues = FeatureValues<RetakeSchema>;

export type RetakeContext = VideoInputContext;

export const RETAKE_MIN_DURATION_SECONDS = 2;
/** Longest selection one retake regenerates, at every resolution. */
export const RETAKE_MAX_DURATION_SECONDS = 10;
/** Packaged example clip is 8s; the demo range is 5s through the end. */
export const RETAKE_EXAMPLE_START_SECONDS = 5;
export const RETAKE_EXAMPLE_DURATION_SECONDS = 3;
export const RETAKE_EXAMPLE_PROMPT =
  "The woman turns her back to the camera and slowly walks away, her face showing a worried expression.";
export const RETAKE_RANGE_LABEL = "Select the part you want to fix";
export const RETAKE_VIDEO_RANGE = {
  startDataKey: "startTime",
  durationDataKey: "duration",
  minDuration: RETAKE_MIN_DURATION_SECONDS,
  maxDuration: RETAKE_MAX_DURATION_SECONDS,
  label: RETAKE_RANGE_LABEL,
} as const;

export const RETAKE_DEFAULTS: RetakeValues = {
  video: null,
  prompt: RETAKE_EXAMPLE_PROMPT,
  model: "ltx-2.5-fast",
  resolution: "original",
  startTime: RETAKE_EXAMPLE_START_SECONDS,
  duration: RETAKE_EXAMPLE_DURATION_SECONDS,
};

export function retakeModelOptions(
  specs: RetakeContext["specs"],
): FieldOption<OfferingId>[] {
  return capabilityModelOptions(specs, "retake");
}

export function hasAdvertisedRetake(specs: RetakeContext["specs"]): boolean {
  return hasAdvertisedCapability(specs, "retake");
}

function resolveRetakeModel(
  model: unknown,
  context: RetakeContext,
): OfferingId {
  return resolveCapableModel(model, context.specs, "retake", RETAKE_DEFAULTS.model);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function regionBounds(context: RetakeContext, startTime: number) {
  const videoLen = context.videoDurationSeconds ?? 0;
  const maxStart = Math.max(0, videoLen - RETAKE_MIN_DURATION_SECONDS);
  const start = clamp(startTime, 0, maxStart);
  const maxDuration = Math.max(
    RETAKE_MIN_DURATION_SECONDS,
    Math.min(RETAKE_MAX_DURATION_SECONDS, videoLen - start),
  );
  return { start, maxStart, maxDuration };
}

export function fromGeneration(
  spec: unknown,
  context: RetakeContext = {
    specs: null,
    seed: null,
    videoDurationSeconds: null,
    videoDurationPending: false,
    videoWidth: null,
    videoHeight: null,
    videoFps: null,
  },
): RetakeValues | null {
  const params = readGenerationParams(spec);
  const inputs = readVideoGenerationInput(spec);
  if (!params || !inputs) return null;
  return {
    video: inputs.video,
    prompt: readPrompt(spec),
    model: restoreOfferingId(params.model, context.specs) ?? RETAKE_DEFAULTS.model,
    resolution: editResolutionKeyFromParams(params),
    startTime:
      typeof params.startTime === "number" ? params.startTime : RETAKE_DEFAULTS.startTime,
    duration:
      typeof params.duration === "number" ? params.duration : RETAKE_DEFAULTS.duration,
  };
}

function videoLengthIssue(
  values: RetakeValues,
  context: RetakeContext,
): ValidationIssue | null {
  if (values.video == null) return null;
  if (context.videoDurationPending) return null;
  if (context.videoDurationSeconds == null) {
    return videoLengthUnknownIssue();
  }
  if (context.videoDurationSeconds + 0.1 < RETAKE_MIN_DURATION_SECONDS) {
    return {
      id: "video-shorter-than-minimum",
      fieldId: "video",
      message: `This video is shorter than ${RETAKE_MIN_DURATION_SECONDS}s. Replace it to continue.`,
      alwaysRevealed: true,
    };
  }
  if (
    values.startTime + values.duration >
    context.videoDurationSeconds + 0.1
  ) {
    return {
      id: "retake-selection-out-of-range",
      fieldId: "video",
      message: "Selection is outside the usable video range",
      alwaysRevealed: true,
    };
  }
  return null;
}

export function validateRetake(
  values: RetakeValues,
  context: RetakeContext,
): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  if (values.video == null) {
    issues.push(videoRequiredIssue());
  } else {
    const lengthIssue = videoLengthIssue(values, context);
    if (lengthIssue) issues.push(lengthIssue);
  }
  return issues;
}

export function toCreateBody(
  values: RetakeValues,
  context: RetakeContext = {
    specs: null,
    seed: null,
    videoDurationSeconds: null,
    videoDurationPending: false,
    videoWidth: null,
    videoHeight: null,
    videoFps: null,
  },
): CreateRetakeRequest {
  if (values.video == null) {
    throw new Error("Video is required.");
  }
  return {
    contract_version: 1,
    inputs: { video: { assetId: values.video.assetId } },
    params: {
      prompt: values.prompt.trim(),
      model: values.model,
      startTime: values.startTime,
      duration: roundDurationSeconds(values.duration),
      mode: RETAKE_CREATE_MODE,
      promptProvenance: "typed",
      ...editResolutionParam(values, context),
    },
  };
}

function editResolutionParam(
  values: RetakeValues,
  context: RetakeContext,
): { resolution: { width: number; height: number } } | Record<string, never> {
  const resolution = editResolutionSize(
    values.resolution,
    context.videoWidth,
    context.videoHeight,
  );
  return resolution ? { resolution } : {};
}

export function buildRetakeForm(
  _values: RetakeValues,
  context: RetakeContext,
): FeatureFormModel<RetakeSchema> {
  const resolutionOptions = editResolutionFieldOptions(
    context.videoWidth,
    context.videoHeight,
    context.specs?.low_performance_machine,
  );
  return {
    fields: [
      {
        kind: "video-asset",
        id: "video",
        label: "Video",
        dataKey: "video",
      },
      {
        kind: "textarea",
        id: "prompt",
        label: "Prompt",
        dataKey: "prompt",
        placeholder: "Describe what should change in this region...",
      },
      ...(resolutionOptions.length > 0
        ? [
            {
              kind: "options" as const,
              id: "resolution",
              label: "Resolution",
              dataKey: "resolution" as const,
              forcePicker: true,
              options: resolutionOptions,
            },
          ]
        : []),
      {
        kind: "options",
        id: "model",
        label: "Model",
        dataKey: "model",
        forcePicker: true,
        options: retakeModelOptions(context.specs),
      },
    ],
  };
}

export function applyRetakeChange(
  values: RetakeValues,
  change: FeatureFieldChange<RetakeSchema>,
): RetakeValues {
  return applyFeatureFieldChange(values, change);
}

export function normalizeRetake(
  values: RetakeValues,
  context: RetakeContext,
): RetakeValues {
  const model = resolveRetakeModel(values.model, context);
  const resolution = normalizeEditResolution(
    values.resolution,
    context.videoWidth,
    context.videoHeight,
  );
  // Unknown length is not 0s. Clamping against 0 collapses the 5s-to-end example
  // to a 2s window at the start, then sticks after the duration loads.
  if (context.videoDurationSeconds == null) {
    const cappedDuration = Math.min(values.duration, RETAKE_MAX_DURATION_SECONDS);
    if (
      model === values.model &&
      resolution === values.resolution &&
      cappedDuration === values.duration
    ) {
      return values;
    }
    return { ...values, model, resolution, duration: cappedDuration };
  }
  const { start, maxDuration } = regionBounds(context, values.startTime);
  const duration = clamp(
    values.duration,
    RETAKE_MIN_DURATION_SECONDS,
    maxDuration,
  );
  if (
    model === values.model &&
    resolution === values.resolution &&
    start === values.startTime &&
    duration === values.duration
  ) {
    return values;
  }
  return { ...values, model, resolution, startTime: start, duration };
}

export function seedRetakeValues(context: RetakeContext): RetakeValues {
  return {
    ...withDefaultOffering(context.specs, { ...RETAKE_DEFAULTS }),
    video: context.seed,
  };
}

export function resetRetakeValues(context: RetakeContext): RetakeValues {
  return withDefaultOffering(context.specs, { ...RETAKE_DEFAULTS });
}

export const retakeDefinition: FeatureDefinition<
  RetakeSchema,
  CreateRetakeRequest,
  RetakeContext,
  "retake"
> = {
  id: "retake",
  title: "Retake",
  defaults: RETAKE_DEFAULTS,
  initialValues: seedRetakeValues,
  resetValues: resetRetakeValues,
  form: buildRetakeForm,
  applyChange: applyRetakeChange,
  normalize: normalizeRetake,
  fromGeneration,
  validate: validateRetake,
  toCreateBody,
  isReady: (values, context) => {
    if (values.video == null) return false;
    if (context.videoDurationPending) return false;
    return videoLengthIssue(values, context) == null;
  },
  isUnavailable: (_values, context) => !hasAdvertisedRetake(context.specs),
};
