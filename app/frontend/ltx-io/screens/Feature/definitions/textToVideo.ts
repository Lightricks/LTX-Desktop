import type { components } from "../../../../generated/backend-openapi";
import { buildMemoryMeterModel } from "../../../../lib/memory-meter.ts";
import {
  effectivePassShape,
  isForcedPassShape,
  withVisiblePassShape,
  type ForcedPassShape,
} from "../../../../lib/pass-shape.ts";
import {
  isVideoGenerationAspectRatio,
  type VideoGenerationAspectRatio,
  type VideoGenerationDuration,
  type VideoGenerationFps,
  type VideoGenerationResolution,
} from "../../../../lib/video-generation-model-specs.ts";
import { readGenerationParams } from "../../../lib/resultsFeedModel.ts";
import {
  clampVideoFields,
  type VideoFeatureContext,
} from "../../../lib/videoFieldPolicy.ts";
import {
  createPassShapeFields,
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

import { changePickerSettings, pickerDefault, pickerReady, pickerSettingsFields, pickerUnavailable } from "../../../lib/videoPickerControls.ts";
import { isExternalModel, pickerGrid, rendererIssue, restoreVideoPickerModel, selectedEngine, wireModel, type VideoPickerModel } from "../../../lib/amdVideoPicker.ts";

type CreateTextToVideoRequest = components["schemas"]["CreateTextToVideoRequest"];

export type TextToVideoSchema = {
  prompt: { kind: "textarea" };
  negativePrompt: { kind: "textarea" };
  model: { kind: "options"; value: VideoPickerModel };
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
  return pickerDefault(TEXT_TO_VIDEO_DEFAULTS, context)
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
    model: restoreVideoPickerModel(params, context?.specs),
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
  if (isExternalModel(values.model)) {
    const issue = rendererIssue(values.model, context?.amd);
    if (issue) issues.push(issue);
    return issues;
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
      model: wireModel(values.model),
      ...selectedEngine(values.model, context.amd),
      aspectRatio: values.aspectRatio,
      resolution: values.resolution,
      duration: values.duration,
      fps: values.fps,
      cameraMotion: "none",
      negativePrompt: isExternalModel(values.model) ? "" : values.negativePrompt.trim(),
      loras: [],
      promptProvenance: "typed",
      passShape: isExternalModel(values.model) ? "auto" : effectivePassShape(context.specs, values),
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
      negativePromptField<TextToVideoSchema>({
        disabledReason: isExternalModel(values.model)
          ? "This local workflow uses only the positive prompt. Choose native LTX to apply your negative prompt."
          : undefined,
      }),
      ...pickerSettingsFields<TextToVideoSchema>(values, context, {
        memoryMeter: true,
      }),
      ...createPassShapeFields(isExternalModel(values.model) ? null : values, context.specs),
    ],
    memory: isExternalModel(values.model) ? null : buildMemoryMeterModel(context.specs, values, {
      fixFields: ["passShape", "duration", "resolution"],
    }),
  };
}

export function applyTextToVideoChange(
  values: TextToVideoValues,
  change: FeatureFieldChange<TextToVideoSchema>,
  context: VideoFeatureContext,
): TextToVideoValues {
  return changePickerSettings<TextToVideoSchema>(
    values,
    change,
    context,
    pickerDefault(TEXT_TO_VIDEO_DEFAULTS, context),
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
      { ...values, model: wireModel(values.model) },
      pickerGrid(context.specs, values.model),
      pickerDefault(TEXT_TO_VIDEO_DEFAULTS, context),
    );
    if (isExternalModel(values.model)) return { ...clamped, passShape: null, model: values.model };
    return withVisiblePassShape(context.specs, clamped, clamped);
  },
  fromGeneration,
  validate: (values, context) => validateTextToVideo(values, context),
  toCreateBody,
  isReady: (values, context) =>
    pickerReady(values, context),
  isUnavailable: (values, context) =>
    pickerUnavailable(values, context),
};
