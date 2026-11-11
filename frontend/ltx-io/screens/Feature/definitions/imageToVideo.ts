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
import { negativePromptField, readNegativePrompt } from "./negativePrompt.ts";
import { readAssetRef } from "./videoFeature.ts";

type CreateImageToVideoRequest = components["schemas"]["CreateImageToVideoRequest"];

export type ImageToVideoAspectRatio = VideoFeatureAspectRatio;

export type ImageToVideoSchema = {
  prompt: { kind: "textarea" };
  negativePrompt: { kind: "textarea" };
  model: { kind: "options"; value: OfferingId };
  aspectRatio: { kind: "options"; value: ImageToVideoAspectRatio };
  resolution: { kind: "options"; value: VideoGenerationResolution };
  duration: { kind: "options"; value: VideoGenerationDuration };
  fps: { kind: "options"; value: VideoGenerationFps };
  passShape: { kind: "options"; value: ForcedPassShape | null };
  startFrame: { kind: "image-asset"; value: AssetRef | null };
  endFrame: { kind: "image-asset"; value: AssetRef | null };
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
};

function isAspectRatio(value: unknown): value is ImageToVideoAspectRatio {
  return value === "auto" || isVideoGenerationAspectRatio(value);
}

function readGenerationInputs(
  spec: unknown,
): { startFrame: AssetRef; endFrame: AssetRef | null } | null {
  if (!spec || typeof spec !== "object") return null;
  const inputs = (spec as { inputs?: unknown }).inputs;
  if (!inputs || typeof inputs !== "object" || Array.isArray(inputs)) {
    return null;
  }
  const startFrame = readAssetRef(
    (inputs as { startFrame?: unknown }).startFrame,
  );
  if (!startFrame) return null;
  const endFrame = readAssetRef((inputs as { endFrame?: unknown }).endFrame);
  return { startFrame, endFrame };
}

export function fromGeneration(
  spec: unknown,
  context?: VideoFeatureContext,
): ImageToVideoValues | null {
  const params = readGenerationParams(spec);
  const inputs = readGenerationInputs(spec);
  if (!params || !inputs) return null;

  return {
    prompt: typeof params.prompt === "string" ? params.prompt : "",
    negativePrompt: readNegativePrompt(spec),
    model: restoreOfferingId(params.model, context?.specs) ?? IMAGE_TO_VIDEO_DEFAULTS.model,
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
  };
}

export function validateImageToVideo(
  values: ImageToVideoValues,
  context?: ImageToVideoContext,
): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  if (values.startFrame == null) {
    issues.push({
      id: "start-frame-required",
      fieldId: "startFrame",
      message: "Start Frame is required.",
    });
  }
  if (values.endFrame != null && values.startFrame == null) {
    issues.push({
      id: "end-frame-requires-start",
      fieldId: "endFrame",
      message: "End Frame requires a Start Frame.",
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
  values: ImageToVideoValues,
  context: ImageToVideoContext,
): CreateImageToVideoRequest {
  if (values.startFrame == null) {
    throw new Error("Start Frame is required.");
  }
  return {
    contract_version: 1,
    inputs: {
      startFrame: { assetId: values.startFrame.assetId },
      endFrame:
        values.endFrame == null
          ? null
          : { assetId: values.endFrame.assetId },
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

export function buildImageToVideoForm(
  values: ImageToVideoValues,
  context: ImageToVideoContext,
): FeatureFormModel<ImageToVideoSchema> {
  return {
    fields: [
      {
        kind: "image-asset",
        id: "startFrame",
        label: "Start Frame",
        dataKey: "startFrame",
      },
      {
        kind: "image-asset",
        id: "endFrame",
        label: "End Frame",
        dataKey: "endFrame",
      },
      {
        kind: "textarea",
        id: "prompt",
        label: "Prompt",
        dataKey: "prompt",
        placeholder: "The woman sips from a cup of coffee...",
      },
      negativePromptField<ImageToVideoSchema>(),
      ...createVideoSettingsOptionFields<ImageToVideoSchema>(values, context.specs, {
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

export function applyImageToVideoChange(
  values: ImageToVideoValues,
  change: FeatureFieldChange<ImageToVideoSchema>,
  context: ImageToVideoContext,
): ImageToVideoValues {
  return applyVideoSettingsChange(
    values,
    change,
    context.specs,
    withDefaultOffering(context.specs, IMAGE_TO_VIDEO_DEFAULTS),
  );
}

export function seedImageToVideoValues(
  context: ImageToVideoContext,
): ImageToVideoValues {
  return {
    ...withDefaultOffering(context.specs, IMAGE_TO_VIDEO_DEFAULTS),
    startFrame: context.seed,
  };
}

export function resetImageToVideoValues(
  context: ImageToVideoContext,
): ImageToVideoValues {
  return withDefaultOffering(context.specs, IMAGE_TO_VIDEO_DEFAULTS);
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
      values,
      context.specs,
      withDefaultOffering(context.specs, IMAGE_TO_VIDEO_DEFAULTS),
    );
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
    isLocalVideoSelectionReady(values, context.specs),
  isUnavailable: (values, context) =>
    !localVideoHasCompatibleOptions(values, context.specs),
};
