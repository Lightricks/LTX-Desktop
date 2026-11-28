import { isH3Variant } from "../../../../lib/h3-models.ts";
import type { components } from "../../../../generated/backend-openapi";
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
  isVideoGenerationAspectRatio,
  type VideoGenerationDuration,
  type VideoGenerationFps,
  type VideoGenerationResolution,
} from "../../../../lib/video-generation-model-specs.ts";
import { readGenerationParams } from "../../../lib/resultsFeedModel.ts";
import {
  clampVideoFields,
  type VideoFeatureAspectRatio,
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
  type AssetRef,
  type FeatureDefinition,
  type FeatureFieldChange,
  type FeatureFormModel,
  type FeatureValues,
  type ValidationIssue,
} from "../types.ts";
import { negativePromptField, readNegativePrompt } from "./negativePrompt.ts";
import { readAssetRef } from "./videoFeature.ts";

import { changePickerSettings, pickerDefault, pickerReady, pickerSettingsFields, pickerUnavailable } from "../../../lib/videoPickerControls.ts";
import { isExternalModel, pickerGrid, rendererIssue, restoreVideoPickerModel, selectedEngine, wireModel, type VideoPickerModel } from "../../../lib/amdVideoPicker.ts";

type CreateImageToVideoRequest = components["schemas"]["CreateImageToVideoRequest"];

export type ImageToVideoAspectRatio = VideoFeatureAspectRatio;

export type ImageToVideoSchema = {
  prompt: { kind: "textarea" };
  negativePrompt: { kind: "textarea" };
  model: { kind: "options"; value: VideoPickerModel };
  aspectRatio: { kind: "options"; value: ImageToVideoAspectRatio };
  resolution: { kind: "options"; value: VideoGenerationResolution };
  duration: { kind: "options"; value: VideoGenerationDuration };
  fps: { kind: "options"; value: VideoGenerationFps };
  passShape: { kind: "options"; value: ForcedPassShape | null };
  startFrame: { kind: "image-asset"; value: AssetRef | null };
  endFrame: { kind: "image-asset"; value: AssetRef | null };
  referenceImages: { kind: "image-assets"; value: AssetRef[] };
};

export type ImageToVideoValues = FeatureValues<ImageToVideoSchema>;

export type ImageToVideoContext = VideoFeatureContext & {
  seed: AssetRef | null;
  /** The start image's size, for the ratio Auto runs at. Null while it loads. */
  startFrameSize?: ImageSize | null;
};


/** Same copy as studio's I2V gallery example — empty-state form seed. */
export const IMAGE_TO_VIDEO_EXAMPLE_PROMPT = "A young man plays the drums.";

export const IMAGE_TO_VIDEO_EXAMPLE_NEGATIVE_PROMPT =
  "extra drumsticks, floating sticks, deformed hands, extra fingers, warped drum kit, frozen motion, blurry face, watermark";

/**
 * Empty state: example prompt + downloaded Fast offering at 540p / 24fps / 8s / Auto.
 * Clamp allows the local short durations, down to 2s.
 */
export const IMAGE_TO_VIDEO_DEFAULTS: ImageToVideoValues = {
  prompt: IMAGE_TO_VIDEO_EXAMPLE_PROMPT,
  negativePrompt: IMAGE_TO_VIDEO_EXAMPLE_NEGATIVE_PROMPT,
  model: "ltx-2.5-fast",
  aspectRatio: "auto",
  resolution: "540p",
  duration: 8,
  fps: 24,
  passShape: null,
  startFrame: null,
  endFrame: null,
  referenceImages: [],
};

function isAspectRatio(value: unknown): value is ImageToVideoAspectRatio {
  return value === "auto" || isVideoGenerationAspectRatio(value);
}

function readGenerationInputs(
  spec: unknown,
): { startFrame: AssetRef | null; endFrame: AssetRef | null; referenceImages: AssetRef[] } | null {
  if (!spec || typeof spec !== "object") return null;
  const inputs = (spec as { inputs?: Record<string, unknown> }).inputs;
  if (!inputs || typeof inputs !== "object" || Array.isArray(inputs)) return null;
  const startFrame = readAssetRef(inputs.startFrame);
  const endFrame = readAssetRef(inputs.endFrame);
  if (!startFrame && !endFrame) return null;
  const referenceImages = Array.isArray(inputs.referenceImages)
    ? inputs.referenceImages.map(readAssetRef).filter((asset): asset is AssetRef => asset !== null) : [];
  return { startFrame, endFrame, referenceImages };
}

export function fromGeneration(
  spec: unknown,
  context?: VideoFeatureContext,
): ImageToVideoValues | null {
  const params = readGenerationParams(spec);
  const inputs = readGenerationInputs(spec);
  if (!params || !inputs) return null;
  if (!inputs.startFrame && params.amdFastEngine !== "h3-fl2va" && params.amdFastEngine !== "h3-fl2va-distill") return null;

  return {
    prompt: typeof params.prompt === "string" ? params.prompt : "",
    negativePrompt: readNegativePrompt(spec),
    model: restoreVideoPickerModel(params, context?.specs),
    aspectRatio: isAspectRatio(params.aspectRatio)
      ? params.aspectRatio
      : IMAGE_TO_VIDEO_DEFAULTS.aspectRatio,
    resolution: isSupportedVideoResolution(params.resolution)
      ? params.resolution
      : IMAGE_TO_VIDEO_DEFAULTS.resolution,
    duration: isSupportedVideoDuration(params.duration)
      ? params.duration
      : IMAGE_TO_VIDEO_DEFAULTS.duration,
    fps: isSupportedVideoFps(params.fps)
      ? params.fps
      : IMAGE_TO_VIDEO_DEFAULTS.fps,
    passShape: isForcedPassShape(params.passShape) ? params.passShape : null,
    startFrame: inputs.startFrame,
    endFrame: inputs.endFrame,
    referenceImages: inputs.referenceImages,
  };
}

export function validateImageToVideo(
  values: ImageToVideoValues,
  context?: ImageToVideoContext,
): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  const h3 = isH3Variant(values.model);
  const refs = values.model === "h3-ref2va";
  const endOnly = h3 && !refs && values.endFrame != null;
  if (values.startFrame == null && !endOnly) {
    issues.push({
      id: "start-frame-required",
      fieldId: "startFrame",
      message: refs ? "Add a reference image." : h3 ? "Add a start or end frame." : "Start Frame is required.",
    });
  }
  if (!h3 && values.endFrame != null && values.startFrame == null) {
    issues.push({
      id: "end-frame-requires-start",
      fieldId: "endFrame",
      message: "End Frame requires a Start Frame.",
    });
  }
  if (isExternalModel(values.model)) {
    const issue = rendererIssue(values.model, context?.amd);
    if (issue) issues.push(issue);
    if (h3 && !values.prompt.trim()) issues.push({ id: "h3-prompt-required", fieldId: "prompt", message: "Enter a prompt for MiniMax H3." });
    if (refs && (values.referenceImages?.length ?? 0) > 8) issues.push({ id: "h3-reference-limit", fieldId: "referenceImages", message: "Use up to nine reference images in total." });
    if (!h3 && values.endFrame) issues.push({ id: "comfy-end-frame", fieldId: "endFrame", message: "Remove the End Frame or choose a native LTX model.", alwaysRevealed: true });
    return issues;
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
  values: ImageToVideoValues,
  context: ImageToVideoContext,
): CreateImageToVideoRequest {
  const refs = values.model === "h3-ref2va";
  if (!values.startFrame && (!isH3Variant(values.model) || refs || !values.endFrame)) {
    throw new Error(refs ? "Add a reference image." : "Add a start frame or an H3 end frame.");
  }
  return {
    contract_version: 1,
    inputs: {
      startFrame: values.startFrame == null ? null : { assetId: values.startFrame.assetId },
      ...(refs && { referenceImages: (values.referenceImages ?? []).map(({ assetId }) => ({ assetId })) }),
      endFrame:
        values.endFrame == null || refs
          ? null
          : { assetId: values.endFrame.assetId },
    },
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
      promptProvenance: "typed",
      passShape: isExternalModel(values.model) ? "auto" : passShapeToSend(context.specs, values, context.startFrameSize),
    },
  };
}

export function buildImageToVideoForm(
  values: ImageToVideoValues,
  context: ImageToVideoContext,
): FeatureFormModel<ImageToVideoSchema> {
  const refs = values.model === "h3-ref2va";
  return {
    fields: [
      {
        kind: "image-asset",
        id: "startFrame",
        label: refs ? "Reference Image 1" : "Start Frame",
        ...(refs && { tooltip: "H3 uses this image as a visual reference, independent of the first frame." }),
        dataKey: "startFrame",
      },
      refs ? {
        kind: "image-assets", id: "referenceImages", label: "Additional reference images",
        dataKey: "referenceImages", maxCount: 8,
        hint: "Add up to eight more images. Describe subjects and actions in the prompt using <Picture 1>, <Picture 2>, etc.",
      } : {
        kind: "image-asset",
        id: "endFrame",
        label: "End Frame",
        dataKey: "endFrame",
        ...(isH3Variant(values.model) && { tooltip: "Anchor the first frame, the last frame, or both." }),
      },
      {
        kind: "textarea",
        id: "prompt",
        label: "Prompt",
        dataKey: "prompt",
        placeholder: "The woman sips from a cup of coffee...",
      },
      negativePromptField<ImageToVideoSchema>({
        disabledReason: isExternalModel(values.model)
          ? "This local workflow uses only the positive prompt. Choose native LTX to apply your negative prompt."
          : undefined,
      }),
      ...pickerSettingsFields<ImageToVideoSchema>(values, context, {
        includeAuto: true,
        memoryMeter: true,
        startFrameSize: context.startFrameSize,
      }),
      ...createPassShapeFields(
        isExternalModel(values.model) ? null : pricedCell(context.specs, values, context.startFrameSize),
        context.specs,
      ),
    ],
    memory: isExternalModel(values.model) ? null : buildMemoryMeterModel(
      context.specs,
      meterCell(context.specs, values, context.startFrameSize),
      { fixFields: ["passShape", "duration", "resolution"] },
    ),
  };
}

export function applyImageToVideoChange(
  values: ImageToVideoValues,
  change: FeatureFieldChange<ImageToVideoSchema>,
  context: ImageToVideoContext,
): ImageToVideoValues {
  return changePickerSettings<ImageToVideoSchema>(
    values,
    change,
    context,
    pickerDefault(IMAGE_TO_VIDEO_DEFAULTS, context),
  );
}

export function seedImageToVideoValues(
  context: ImageToVideoContext,
): ImageToVideoValues {
  return {
    ...pickerDefault(IMAGE_TO_VIDEO_DEFAULTS, context),
    startFrame: context.seed,
  };
}

export function resetImageToVideoValues(
  context: ImageToVideoContext,
): ImageToVideoValues {
  return pickerDefault(IMAGE_TO_VIDEO_DEFAULTS, context);
}

export const imageToVideoDefinition: FeatureDefinition<
  ImageToVideoSchema,
  CreateImageToVideoRequest,
  ImageToVideoContext,
  "image-to-video"
> = {
  id: "image-to-video",
  title: "Image to Video",
  defaults: IMAGE_TO_VIDEO_DEFAULTS,
  initialValues: seedImageToVideoValues,
  resetValues: resetImageToVideoValues,
  form: buildImageToVideoForm,
  applyChange: applyImageToVideoChange,
  normalize: (values, context) => {
    const clamped = clampVideoFields(
      { ...values, model: wireModel(values.model) },
      pickerGrid(context.specs, values.model),
      pickerDefault(IMAGE_TO_VIDEO_DEFAULTS, context),
    );
    if (isExternalModel(values.model)) return { ...clamped, passShape: null, model: values.model };
    return withVisiblePassShape(
      context.specs,
      clamped,
      pricedCell(context.specs, clamped, context.startFrameSize),
    );
  },
  fromGeneration,
  validate: validateImageToVideo,
  toCreateBody,
  isReady: (values, context) =>
    pickerReady(values, context),
  isUnavailable: (values, context) =>
    pickerUnavailable(values, context),
};
