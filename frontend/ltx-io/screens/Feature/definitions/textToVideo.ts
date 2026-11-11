import type { components } from "../../../../generated/backend-openapi";
import { buildMemoryMeterModel } from "../../../../lib/memory-meter.ts";
import {
  effectivePassShape,
  isForcedPassShape,
  withVisiblePassShape,
  type ForcedPassShape,
} from "../../../../lib/pass-shape.ts";
import {
  restoreOfferingId,
  withDefaultOffering,
  type OfferingId,
  isVideoGenerationAspectRatio,
  type VideoGenerationAspectRatio,
  type VideoGenerationDuration,
  type VideoGenerationFps,
  type VideoGenerationResolution,
} from "../../../../lib/video-generation-model-specs.ts";
import { readGenerationParams } from "../../../lib/resultsFeedModel.ts";
import {
  clampVideoFields,
  isLocalVideoSelectionReady,
  localVideoHasCompatibleOptions,
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
  type FeatureValues,
  type ValidationIssue,
} from "../types.ts";
import { negativePromptField, readNegativePrompt } from "./negativePrompt.ts";

type CreateTextToVideoRequest = components["schemas"]["CreateTextToVideoRequest"];

export type TextToVideoSchema = {
  prompt: { kind: "textarea" };
  negativePrompt: { kind: "textarea" };
  model: { kind: "options"; value: OfferingId };
  aspectRatio: { kind: "options"; value: VideoGenerationAspectRatio };
  resolution: { kind: "options"; value: VideoGenerationResolution };
  duration: { kind: "options"; value: VideoGenerationDuration };
  fps: { kind: "options"; value: VideoGenerationFps };
  passShape: { kind: "options"; value: ForcedPassShape | null };
};

export type TextToVideoValues = FeatureValues<TextToVideoSchema>;

/** Same copy as studio's T2V gallery example — empty-state form seed. */
export const TEXT_TO_VIDEO_EXAMPLE_PROMPT =
  "A dalmatian dog running fast along a sandy beach in dynamic galloping stride, head forward and alert. Soft overcast daylight with cool blue atmospheric tone. Calm water on the left, dense pine forest backdrop, vintage cream and red camper van in distance. Cinematic camera following alongside, shallow depth of field with sharp dog and softer background. Soft hazy lighting with subtle shadows on sand. Wind motion through fur, nostalgic cozy mood. Sound of pounding paws hitting sand rhythmically, wind rushing past, distant gentle waves lapping the shore, muffled forest ambience in the background.";

export const TEXT_TO_VIDEO_EXAMPLE_NEGATIVE_PROMPT =
  "extra legs, deformed paws, merged limbs, blurry dog, camera shake, harsh midday glare, subtitles, watermark";

export const TEXT_TO_VIDEO_DEFAULTS: TextToVideoValues = {
  prompt: TEXT_TO_VIDEO_EXAMPLE_PROMPT,
  negativePrompt: TEXT_TO_VIDEO_EXAMPLE_NEGATIVE_PROMPT,
  model: "ltx-2.5-fast",
  aspectRatio: "16:9",
  resolution: "540p",
  duration: 8,
  fps: 24,
  passShape: null,
};

export function seedTextToVideoValues(
  context: VideoFeatureContext,
): TextToVideoValues {
  return withDefaultOffering(context.specs, TEXT_TO_VIDEO_DEFAULTS)
}

export function fromGeneration(
  spec: unknown,
  context?: VideoFeatureContext,
): TextToVideoValues | null {
  const params = readGenerationParams(spec);
  if (!params || typeof params.prompt !== "string") return null;

  return {
    prompt: params.prompt,
    negativePrompt: readNegativePrompt(spec),
    model: restoreOfferingId(params.model, context?.specs) ?? TEXT_TO_VIDEO_DEFAULTS.model,
    aspectRatio: isVideoGenerationAspectRatio(params.aspectRatio)
      ? params.aspectRatio
      : TEXT_TO_VIDEO_DEFAULTS.aspectRatio,
    resolution: isSupportedVideoResolution(params.resolution)
      ? params.resolution
      : TEXT_TO_VIDEO_DEFAULTS.resolution,
    duration: isSupportedVideoDuration(params.duration)
      ? params.duration
      : TEXT_TO_VIDEO_DEFAULTS.duration,
    fps: isSupportedVideoFps(params.fps) ? params.fps : TEXT_TO_VIDEO_DEFAULTS.fps,
    passShape: isForcedPassShape(params.passShape) ? params.passShape : null,
  };
}

export function validateTextToVideo(
  values: TextToVideoValues,
  context?: VideoFeatureContext,
): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  if (values.prompt.trim().length === 0) {
    issues.push({
      id: "prompt-required",
      fieldId: "prompt",
      message: "Prompt is required.",
    });
  }
  issues.push(...fastBudgetIssues(context?.specs, values));
  return issues;
}

export function toCreateBody(
  values: TextToVideoValues,
  context: VideoFeatureContext,
): CreateTextToVideoRequest {
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
      loras: [],
      promptProvenance: "typed",
      passShape: effectivePassShape(context.specs, values),
    },
  };
}

export function buildTextToVideoForm(
  values: TextToVideoValues,
  context: VideoFeatureContext,
): FeatureFormModel<TextToVideoSchema> {
  return {
    fields: [
      {
        kind: "textarea",
        id: "prompt",
        label: "Prompt",
        dataKey: "prompt",
        placeholder: "The woman sips from a cup of coffee...",
      },
      negativePromptField<TextToVideoSchema>(),
      ...createVideoSettingsOptionFields<TextToVideoSchema>(values, context.specs, {
        memoryMeter: true,
      }),
      ...createPassShapeFields(values, context.specs),
    ],
    memory: buildMemoryMeterModel(context.specs, values, {
      fixFields: ["passShape", "duration", "resolution"],
    }),
  };
}

export function applyTextToVideoChange(
  values: TextToVideoValues,
  change: FeatureFieldChange<TextToVideoSchema>,
  context: VideoFeatureContext,
): TextToVideoValues {
  return applyVideoSettingsChange(
    values,
    change,
    context.specs,
    withDefaultOffering(context.specs, TEXT_TO_VIDEO_DEFAULTS),
  );
}

export const textToVideoDefinition: FeatureDefinition<
  TextToVideoSchema,
  CreateTextToVideoRequest,
  VideoFeatureContext,
  "text-to-video"
> = {
  id: "text-to-video",
  title: "Text to Video",
  defaults: TEXT_TO_VIDEO_DEFAULTS,
  initialValues: seedTextToVideoValues,
  resetValues: seedTextToVideoValues,
  form: buildTextToVideoForm,
  applyChange: applyTextToVideoChange,
  normalize: (values, context) => {
    const clamped = clampVideoFields(
      values,
      context.specs,
      withDefaultOffering(context.specs, TEXT_TO_VIDEO_DEFAULTS),
    );
    return withVisiblePassShape(context.specs, clamped, clamped);
  },
  fromGeneration,
  validate: (values, context) => validateTextToVideo(values, context),
  toCreateBody,
  isReady: (values, context) =>
    isLocalVideoSelectionReady(values, context.specs),
  isUnavailable: (values, context) =>
    !localVideoHasCompatibleOptions(values, context.specs),
};
