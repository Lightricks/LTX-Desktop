import type { components } from "../../../../generated/backend-openapi";
import {
  restoreOfferingId,
  withDefaultOffering,
  type OfferingId,
} from "../../../../lib/video-generation-model-specs.ts";
import { preflightEstimatedTimeText } from "../../../lib/editPreflight.ts";
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
import { negativePromptField } from "./negativePrompt.ts";
import {
  capabilityModelOptions,
  editPreflightIssues,
  editResolutionFieldOptions,
  editResolutionSize,
  editResolutionKeyFromParams,
  type EditResolutionKey,
  hasAdvertisedCapability,
  normalizeEditResolution,
  readNegativePrompt,
  readPrompt,
  readVideoGenerationInput,
  resolveCapableModel,
  resolvePreflightResolution,
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
  negativePrompt: { kind: "textarea" };
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
export const RETAKE_EXAMPLE_NEGATIVE_PROMPT =
  "smiling, laughing, facing camera, walking forward, abrupt cut, seam jump, warped face, flicker";
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
  negativePrompt: RETAKE_EXAMPLE_NEGATIVE_PROMPT,
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

/** Wire values create enforces. Validation reads these, never raw form state. */
function retakeWireValues(values: RetakeValues): {
  startTime: number;
  duration: number;
} {
  return {
    // 0.1s grid like duration: the trim is continuous and create prices
    // the wire values, so rounding the wire start up to the 0.5s preflight
    // grid over-refuses tail selections (6.3s on an 8.3s clip would wire
    // as 6.5s + 2.0s = 8.5s > 8.4s). Only the preflight *request* rounds
    // to 0.5s (RetakeFeatureScreen, to keep drags from refetching).
    startTime: roundDurationSeconds(values.startTime),
    duration: roundDurationSeconds(values.duration),
  };
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
    negativePrompt: readNegativePrompt(spec),
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
  const wire = retakeWireValues(values);
  if (wire.startTime + wire.duration > context.videoDurationSeconds + 0.1) {
    return {
      id: "retake-selection-out-of-range",
      fieldId: "video",
      message: "Selection is outside the usable video range",
      alwaysRevealed: true,
    };
  }
  return null;
}

function preflightIssues(
  values: RetakeValues,
  context: RetakeContext,
): ValidationIssue[] {
  return editPreflightIssues({
    video: values.video,
    preflight: context.preflight,
    resolution: values.resolution,
    duration: retakeWireValues(values).duration,
    minDuration: RETAKE_MIN_DURATION_SECONDS,
    resolutionIssueId: "retake-resolution-too-large",
    durationIssueId: "retake-selection-too-long",
    durationFieldId: "video",
  });
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
  issues.push(...preflightIssues(values, context));
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
  const wire = retakeWireValues(values);
  return {
    contract_version: 1,
    inputs: { video: { assetId: values.video.assetId } },
    params: {
      prompt: values.prompt.trim(),
      negativePrompt: values.negativePrompt.trim(),
      model: values.model,
      startTime: wire.startTime,
      duration: wire.duration,
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
  values: RetakeValues,
  context: RetakeContext,
): FeatureFormModel<RetakeSchema> {
  const resolutionOptions = editResolutionFieldOptions(
    context.videoWidth,
    context.videoHeight,
    context.specs?.low_performance_machine,
    { state: context.preflight, minDuration: RETAKE_MIN_DURATION_SECONDS },
    context.highResVideoToVideo,
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
      negativePromptField<RetakeSchema>(),
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
    estimatedTimeText:
      values.video == null
        ? null
        : preflightEstimatedTimeText(
            context.preflight,
            values.resolution,
            retakeWireValues(values).duration,
          ),
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
  const resolution = resolvePreflightResolution(
    normalizeEditResolution(
      values.resolution,
      context.videoWidth,
      context.videoHeight,
      context.highResVideoToVideo,
    ),
    context,
    RETAKE_MIN_DURATION_SECONDS,
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
    if (videoLengthIssue(values, context) != null) return false;
    return preflightIssues(values, context).every(
      (issue) => issue.blocksGenerate === false,
    );
  },
  isUnavailable: (_values, context) => !hasAdvertisedRetake(context.specs),
};
