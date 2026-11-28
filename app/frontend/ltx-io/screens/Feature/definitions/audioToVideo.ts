import type { components } from "../../../../generated/backend-openapi";
import { type SelectionRisk, worseRisk } from "../../../../lib/budget-risk.ts";
import { closestVideoAspectRatio } from "../../../../lib/closest-video-aspect-ratio.ts";
import { buildMemoryMeterModel, optionMemory } from "../../../../lib/memory-meter.ts";
import {
  effectivePassShape,
  isForcedPassShape,
  withVisiblePassShape,
  type ForcedPassShape,
  type PassShapeSelection,
} from "../../../../lib/pass-shape.ts";
import {
  type OfferingId,
  type VideoGenerationAspectRatio,
  fastSelectionRisk,
  isVideoGenerationAspectRatio,
  orderAspectRatiosForPicker,
  restoreOfferingId,
  untrainedAspectRatioWarning,
  withDefaultOffering,
} from "../../../../lib/video-generation-model-specs.ts";
import { videoGenerationResolutionLabel } from "../../../../lib/video-resolution.ts";
import { generationResolutionWarning } from "../../../lib/lowPerformanceWarning.ts";
import {
  A2V_FPS,
  A2V_MIN_AUDIO_SECONDS,
  A2V_RESOLUTION,
  type A2vAudioFit,
  type A2vResolution,
  a2vAspectRatios,
  a2vDisabledAspectsForOffering,
  a2vDisabledResolutionsForOffering,
  a2vDurationForFrames,
  a2vResolutionsForOffering,
  a2vTrimCapSeconds,
  advertisedA2vDurations,
  hasAdvertisedA2vCell,
  isA2vResolution,
  resolveA2vAudio,
  staticA2vDurations,
  staticA2vLongestCellSeconds,
} from "../../../lib/a2vDurationPolicy.ts";
import { readGenerationParams } from "../../../lib/resultsFeedModel.ts";
import type {
  VideoFeatureAspectRatio,
  VideoFeatureContext,
} from "../../../lib/videoFieldPolicy.ts";
import {
  createPassShapeFields,
  fastBudgetIssues,
  VIDEO_SETTING_RESOLUTIONS,
} from "../../../lib/videoSettingsControls.ts";
import {
  type AssetRef,
  type FeatureDefinition,
  type FeatureFieldChange,
  type FeatureFormModel,
  type FeatureValues,
  type FieldOption,
  type ValidationIssue,
  applyFeatureFieldChange,
} from "../types.ts";
import { negativePromptField, readNegativePrompt } from "./negativePrompt.ts";

import { readAssetRef } from "./videoFeature.ts";

type CreateAudioToVideoRequest = components["schemas"]["CreateAudioToVideoRequest"];

export type AudioToVideoAspectRatio = VideoFeatureAspectRatio;

export type AudioToVideoSchema = {
  startFrame: { kind: "image-asset"; value: AssetRef | null };
  prompt: { kind: "textarea" };
  negativePrompt: { kind: "textarea" };
  audio: { kind: "audio-asset"; value: AssetRef | null };
  model: { kind: "options"; value: OfferingId };
  resolution: { kind: "options"; value: A2vResolution };
  aspectRatio: { kind: "options"; value: AudioToVideoAspectRatio };
  passShape: { kind: "options"; value: ForcedPassShape | null };
};

export type AudioToVideoValues = FeatureValues<AudioToVideoSchema>;

/** Both packaged example slots arrive together. */
export type AudioToVideoSeed = {
  audio: AssetRef;
  startFrame: AssetRef;
};

export type AudioToVideoContext = VideoFeatureContext & {
  seed: AudioToVideoSeed | null;
  /** Length of the accepted (already trimmed) audio asset. Null until it loads. */
  audioDurationSeconds: number | null;
  /**
   * The audio asset is still being fetched, so a null `audioDurationSeconds`
   * means "not yet" rather than "this file has no readable length".
   */
  audioDurationPending: boolean;
  /**
   * Start-frame pixels once that asset has loaded. Auto prices the closest
   * ratio from these, matching the backend. Null while the asset is loading.
   */
  startFrameSize: { width: number; height: number } | null;
};

const AUTO_ASPECT_OPTION: FieldOption<AudioToVideoAspectRatio> = {
  value: "auto",
  label: "Auto",
};

function a2vFit(
  specs: AudioToVideoContext["specs"],
  offering: OfferingId | null | undefined,
  resolution: A2vResolution,
  audioSeconds: number,
  aspectRatio?: string | null,
): A2vAudioFit {
  return resolveA2vAudio({
    audioSeconds,
    keptSeconds: advertisedA2vDurations(specs, offering, resolution, aspectRatio),
    ceilingSeconds: staticA2vDurations(specs, offering, resolution, aspectRatio),
    longestCeilingSeconds: staticA2vLongestCellSeconds(specs, offering, aspectRatio),
  });
}

function explicitAspectOptions(
  context: AudioToVideoContext,
  offering: OfferingId,
  resolution: A2vResolution,
  passShape?: ForcedPassShape | null,
): FieldOption<AudioToVideoAspectRatio>[] {
  const fit =
    context.audioDurationSeconds == null
      ? null
      : a2vFit(context.specs, offering, resolution, context.audioDurationSeconds);
  const effective = fit != null && !fit.pastKept ? fit.heardSeconds : null;
  const priced = pricedA2vDuration(
    context.specs,
    offering,
    resolution,
    context.audioDurationSeconds,
  );
  const disabled = a2vDisabledAspectsForOffering(
    context.specs,
    offering,
    resolution,
    effective,
  );
  return orderAspectRatiosForPicker(
    a2vAspectRatios(context.specs, offering, resolution),
  ).map((value) => {
    const warning = untrainedAspectRatioWarning(value);
    const refused = disabled.find((option) => option.value === value);
    const aspectPriced =
      pricedA2vDuration(
        context.specs,
        offering,
        resolution,
        context.audioDurationSeconds,
        value,
      ) ?? priced;
    const memory = optionMemory(context.specs, {
      model: offering,
      resolution,
      fps: A2V_FPS,
      duration: aspectPriced,
      aspectRatio: value,
      audio: true,
      passShape,
    });
    return {
      value,
      label: value,
      ...(warning != null ? { warning } : {}),
      ...(refused ? { disabledReason: refused.message } : {}),
      ...(memory === undefined ? {} : { memory }),
    };
  });
}

/** Same copy as studio's A2V lipsync placeholder — empty-state form seed. */
export const AUDIO_TO_VIDEO_EXAMPLE_PROMPT =
  "The character speaks while moving through the scene";

export const AUDIO_TO_VIDEO_EXAMPLE_NEGATIVE_PROMPT =
  "closed mouth, out-of-sync lips, warped teeth, distorted jaw, frozen face, robotic movement, subtitles, watermark";

/**
 * Empty state: example prompt plus Auto aspect, which matches the packaged
 * example start image. Normalize rewrites Auto to 16:9 when no image is set.
 */
export const AUDIO_TO_VIDEO_DEFAULTS: AudioToVideoValues = {
  startFrame: null,
  prompt: AUDIO_TO_VIDEO_EXAMPLE_PROMPT,
  negativePrompt: AUDIO_TO_VIDEO_EXAMPLE_NEGATIVE_PROMPT,
  audio: null,
  model: "ltx-2.5-fast",
  resolution: A2V_RESOLUTION,
  aspectRatio: "auto",
  passShape: null,
};

function isAspectRatio(value: unknown): value is AudioToVideoAspectRatio {
  return value === "auto" || isVideoGenerationAspectRatio(value);
}

function readGenerationInputs(
  spec: unknown,
): { audio: AssetRef; startFrame: AssetRef | null } | null {
  if (!spec || typeof spec !== "object") return null;
  const inputs = (spec as { inputs?: unknown }).inputs;
  if (!inputs || typeof inputs !== "object" || Array.isArray(inputs)) {
    return null;
  }
  const audio = readAssetRef((inputs as { audio?: unknown }).audio);
  if (!audio) return null;
  const startFrame = readAssetRef((inputs as { startFrame?: unknown }).startFrame);
  return { audio, startFrame };
}

export function a2vModelOptions(
  specs: AudioToVideoContext["specs"],
): FieldOption<OfferingId>[] {
  return (specs?.downloaded_local_models ?? [])
    .filter(
      (item) => advertisedA2vDurations(specs, item.model, A2V_RESOLUTION).length > 0,
    )
    .map((item) => ({
      value: item.model,
      label: item.spec.display_name,
    }));
}

function a2vResolutionOptions(
  context: AudioToVideoContext,
  offering: OfferingId,
  aspectRatio?: AudioToVideoAspectRatio,
  passShape?: ForcedPassShape | null,
): FieldOption<A2vResolution>[] {
  const specs = context.specs;
  const resolvedAspect =
    aspectRatio != null && aspectRatio !== "auto" ? aspectRatio : null;
  // Kept-then-dropped stays sorted: a kept resolution implies all smaller
  // ones kept, so drops append above keeps in A2V order.
  return [
    ...a2vResolutionsForOffering(specs, offering, resolvedAspect).map((resolution) => {
      const priced = pricedA2vDuration(
        specs,
        offering,
        resolution,
        context.audioDurationSeconds,
        resolvedAspect,
      );
      const memory = optionMemory(specs, {
        model: offering,
        resolution,
        fps: A2V_FPS,
        duration: priced,
        aspectRatio: resolvedAspect,
        audio: true,
        passShape,
      });
      const warning = generationResolutionWarning(resolution, specs?.low_performance_machine);
      return {
        value: resolution,
        label: videoGenerationResolutionLabel(resolution),
        ...(warning != null ? { warning } : {}),
        ...(memory === undefined ? {} : { memory }),
      };
    }),
    ...a2vDisabledResolutionsForOffering(specs, offering, resolvedAspect).map((option) => ({
      value: option.value,
      label: videoGenerationResolutionLabel(option.value),
      disabledReason: option.message,
    })),
  ];
}

/**
 * Smallest kept cell whose frame count covers the heard audio. The backend
 * prices the exact frame count, so this can warn when the queued job stores
 * no risk. It does not price a longer cell than those frames need.
 */
function pricedA2vDuration(
  specs: AudioToVideoContext["specs"],
  offering: OfferingId | null | undefined,
  resolution: A2vResolution,
  audioSeconds: number | null,
  aspectRatio?: string | null,
): number | null {
  if (audioSeconds == null) return null;
  const fit = a2vFit(specs, offering, resolution, audioSeconds, aspectRatio);
  if (fit.pastKept || fit.frames == null) return null;
  return a2vDurationForFrames(
    advertisedA2vDurations(specs, offering, resolution, aspectRatio),
    fit.frames,
  );
}

function resolveA2vModel(model: unknown, context: AudioToVideoContext): OfferingId {
  const options = a2vModelOptions(context.specs);
  const restored = restoreOfferingId(model, context.specs);
  if (restored && options.some((item) => item.value === restored)) {
    return restored;
  }
  return options[0]?.value ?? AUDIO_TO_VIDEO_DEFAULTS.model;
}

function resolveA2vResolution(
  resolution: unknown,
  specs: AudioToVideoContext["specs"],
  offering: OfferingId,
  aspectRatio?: string | null,
): A2vResolution {
  const options = a2vResolutionsForOffering(specs, offering, aspectRatio);
  if (isA2vResolution(resolution) && options.includes(resolution)) {
    return resolution;
  }
  return options[0] ?? A2V_RESOLUTION;
}

/**
 * Client-side A2V frame estimate for the generate gate. Null when the audio
 * has not loaded, is under 2s, exceeds the static ceiling, or the heard
 * frames do not fit a budget-kept cell. A clip past this resolution but
 * inside a longer kept cell is estimated from that cell's max. The server
 * derives and persists the authoritative `numFrames`.
 */
export function a2vAudioNumFrames(
  context: AudioToVideoContext,
  offering?: OfferingId | null,
  resolution: A2vResolution = A2V_RESOLUTION,
  aspectRatio?: string | null,
): number | null {
  if (context.audioDurationSeconds == null) return null;
  const fit = a2vFit(
    context.specs,
    offering,
    resolution,
    context.audioDurationSeconds,
    aspectRatio,
  );
  if (fit.pastKept || fit.frames == null) return null;
  return fit.frames;
}

export function fromGeneration(
  spec: unknown,
  context: AudioToVideoContext = {
    specs: null,
    seed: null,
    audioDurationSeconds: null,
    audioDurationPending: false,
    startFrameSize: null,
  },
): AudioToVideoValues | null {
  const params = readGenerationParams(spec);
  const inputs = readGenerationInputs(spec);
  if (!params || !inputs) return null;
  const model =
    restoreOfferingId(params.model, context.specs) ?? AUDIO_TO_VIDEO_DEFAULTS.model;
  const aspectRatio = isAspectRatio(params.aspectRatio)
    ? params.aspectRatio
    : AUDIO_TO_VIDEO_DEFAULTS.aspectRatio;

  return {
    startFrame: inputs.startFrame,
    prompt: typeof params.prompt === "string" ? params.prompt : "",
    negativePrompt: readNegativePrompt(spec),
    audio: inputs.audio,
    model,
    resolution: resolveA2vResolution(params.resolution, context.specs, model, aspectRatio),
    aspectRatio,
    passShape: isForcedPassShape(params.passShape) ? params.passShape : null,
  };
}

/**
 * Why the attached audio cannot generate, or that this resolution will hear
 * only its opening prefix when a longer cell exists. Blocking cases disable
 * Generate, so they are revealed on sight.
 */
function audioLengthIssue(
  values: AudioToVideoValues,
  context: AudioToVideoContext,
): ValidationIssue | null {
  if (context.audioDurationPending) return null;
  if (context.audioDurationSeconds == null) {
    return {
      id: "audio-duration-unknown",
      fieldId: "audio",
      message: "We couldn't read this audio's length. Replace it to continue.",
      alwaysRevealed: true,
    };
  }
  if (context.audioDurationSeconds < A2V_MIN_AUDIO_SECONDS) {
    return {
      id: "audio-shorter-than-minimum",
      fieldId: "audio",
      message: "This audio is shorter than 2s. Replace it to continue.",
      alwaysRevealed: true,
    };
  }
  const ceiling = staticA2vDurations(
    context.specs,
    values.model,
    values.resolution,
    values.aspectRatio,
  );
  const fit = a2vFit(
    context.specs,
    values.model,
    values.resolution,
    context.audioDurationSeconds,
    values.aspectRatio,
  );
  // A prefix is hearable only when its frames fit a kept cell. A dropped
  // ceiling is a refusal, even when a longer resolution made the backend trim.
  if (
    fit.heardSeconds != null &&
    fit.heardSeconds < context.audioDurationSeconds &&
    !fit.pastKept
  ) {
    return {
      id: "audio-longer-than-resolution",
      fieldId: "audio",
      message: `At ${videoGenerationResolutionLabel(values.resolution)}, the maximum is ${fit.heardSeconds} seconds. We'll use only the first ${fit.heardSeconds}s, or you can choose a lower resolution.`,
      alwaysRevealed: true,
      blocksGenerate: false,
    };
  }
  if (
    fit.pastKept ||
    (fit.heardSeconds == null && fit.keptMaxSeconds == null && ceiling.length > 0)
  ) {
    return {
      id: "audio-past-kept",
      fieldId: "audio",
      message:
        fit.keptMaxSeconds == null
          ? `This clip can't run at ${values.resolution} on this machine.`
          : `At ${values.resolution}, this machine can run up to ${fit.keptMaxSeconds}s. Choose a lower resolution or a shorter clip.`,
      alwaysRevealed: true,
    };
  }
  if (fit.heardSeconds != null) return null;
  const cap = fit.keptMaxSeconds ?? a2vTrimCapSeconds(context.specs, values.model);
  // No cap means the machine advertises no A2V cell at all, which the
  // feature-unavailable message already explains.
  if (cap == null) return null;
  return {
    id: "audio-longer-than-cap",
    fieldId: "audio",
    message: `This audio is longer than ${cap}s at ${videoGenerationResolutionLabel(values.resolution)}. Choose a lower resolution or replace it with a shorter clip.`,
    alwaysRevealed: true,
  };
}

/**
 * Explicit ratios pass through. Auto with no start frame is 16:9, the same
 * fallback the backend uses. Auto with a loaded frame uses that frame's
 * closest ratio. Until the frame loads, the loudest aspect is used so the
 * warning cannot be quieter than the ratio the backend will price.
 */
function aspectForA2vBudget(
  values: AudioToVideoValues,
  context: AudioToVideoContext,
): VideoGenerationAspectRatio {
  if (isVideoGenerationAspectRatio(values.aspectRatio)) return values.aspectRatio;
  if (values.startFrame == null) return "16:9";
  const ratios = a2vAspectRatios(context.specs, values.model, values.resolution);
  const size = context.startFrameSize;
  if (size != null && size.height > 0) {
    return closestVideoAspectRatio(size.width, size.height, ratios) ?? "16:9";
  }
  const priced = pricedA2vDuration(
    context.specs,
    values.model,
    values.resolution,
    context.audioDurationSeconds,
  );
  let chosen: VideoGenerationAspectRatio = "16:9";
  let worst: SelectionRisk | null = null;
  for (const ratio of ratios) {
    const risk = fastSelectionRisk(context.specs, {
      model: values.model,
      resolution: values.resolution,
      fps: A2V_FPS,
      duration: priced,
      aspectRatio: ratio,
      audio: true,
    });
    const next = worseRisk(worst, risk);
    if (next !== worst && risk != null) {
      worst = next;
      chosen = ratio;
    }
  }
  return chosen;
}

export function validateAudioToVideo(
  values: AudioToVideoValues,
  context: AudioToVideoContext,
): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  if (values.audio == null) {
    issues.push({
      id: "audio-required",
      fieldId: "audio",
      message: "Audio is required.",
    });
  } else {
    const lengthIssue = audioLengthIssue(values, context);
    if (lengthIssue) issues.push(lengthIssue);
  }
  if (values.prompt.trim().length === 0 && values.startFrame == null) {
    issues.push({
      id: "prompt-or-start-frame-required",
      fieldId: "prompt",
      message: "Add a prompt or a start image.",
    });
  }
  issues.push(...fastBudgetIssues(context.specs, a2vCellSelection(values, context)));
  return issues;
}

/** The cell the job is priced at, or null until the audio is in: its length is the job's length. */
function a2vCellSelection(
  values: AudioToVideoValues,
  context: AudioToVideoContext,
): PassShapeSelection | null {
  const priced = pricedA2vDuration(
    context.specs,
    values.model,
    values.resolution,
    context.audioDurationSeconds,
    values.aspectRatio,
  );
  if (values.audio == null || priced == null) return null;
  return {
    model: values.model,
    resolution: values.resolution,
    fps: A2V_FPS,
    duration: priced,
    aspectRatio: aspectForA2vBudget(values, context),
    audio: true,
    passShape: values.passShape,
  };
}

function audioToVideoMemory(values: AudioToVideoValues, context: AudioToVideoContext) {
  const selection = a2vCellSelection(values, context);
  if (selection == null) return null;
  return buildMemoryMeterModel(
    context.specs,
    selection,
    {
      fixFields: ["passShape", "resolution"],
      durationAt: (resolution) =>
        isA2vResolution(resolution)
          ? pricedA2vDuration(
              context.specs,
              values.model,
              resolution,
              context.audioDurationSeconds,
              values.aspectRatio,
            )
          : null,
    },
  );
}

export function toCreateBody(
  values: AudioToVideoValues,
  context: AudioToVideoContext,
): CreateAudioToVideoRequest {
  if (values.audio == null) {
    throw new Error("Audio is required.");
  }
  if (
    a2vAudioNumFrames(context, values.model, values.resolution, values.aspectRatio) ==
    null
  ) {
    throw new Error("This audio is longer than this machine can generate.");
  }
  return {
    contract_version: 1,
    inputs: {
      audio: { assetId: values.audio.assetId },
      startFrame:
        values.startFrame == null ? null : { assetId: values.startFrame.assetId },
    },
    params: {
      prompt: values.prompt.trim(),
      model: values.model,
      aspectRatio: values.aspectRatio,
      resolution: values.resolution,
      fps: A2V_FPS,
      cameraMotion: "none",
      negativePrompt: values.negativePrompt.trim(),
      promptProvenance: "typed",
      passShape: passShapeFor(values, context),
    },
  };
}

function passShapeFor(values: AudioToVideoValues, context: AudioToVideoContext) {
  const selection = a2vCellSelection(values, context);
  return selection == null ? "auto" : effectivePassShape(context.specs, selection);
}

export function buildAudioToVideoForm(
  values: AudioToVideoValues,
  context: AudioToVideoContext,
): FeatureFormModel<AudioToVideoSchema> {
  return {
    fields: [
      {
        kind: "image-asset",
        id: "startFrame",
        label: "Start Frame",
        dataKey: "startFrame",
      },
      {
        kind: "textarea",
        id: "prompt",
        label: "Prompt",
        dataKey: "prompt",
        placeholder: "The character speaks while moving through the scene...",
      },
      negativePromptField<AudioToVideoSchema>(),
      {
        kind: "audio-asset",
        id: "audio",
        label: "Audio",
        dataKey: "audio",
      },
      {
        kind: "options",
        id: "model",
        label: "Model",
        dataKey: "model",
        forcePicker: true,
        options: a2vModelOptions(context.specs),
      },
      {
        kind: "options",
        id: "resolution",
        label: "Resolution",
        dataKey: "resolution",
        maxOptionCount: VIDEO_SETTING_RESOLUTIONS.length,
        options: a2vResolutionOptions(
          context,
          values.model,
          values.aspectRatio,
          values.passShape,
        ),
      },
      {
        kind: "options",
        id: "aspectRatio",
        label: "Aspect Ratio",
        dataKey: "aspectRatio",
        // Auto derives the ratio from the start image, so it is only offered
        // while one is set.
        options: [
          ...(values.startFrame == null ? [] : [AUTO_ASPECT_OPTION]),
          ...explicitAspectOptions(
            context,
            values.model,
            values.resolution,
            values.passShape,
          ),
        ],
      },
      ...a2vPassShapeFields(values, context),
    ],
    memory: audioToVideoMemory(values, context),
  };
}

function a2vPassShapeFields(values: AudioToVideoValues, context: AudioToVideoContext) {
  const selection = a2vCellSelection(values, context);
  return selection == null ? [] : createPassShapeFields(selection, context.specs);
}

export function applyAudioToVideoChange(
  values: AudioToVideoValues,
  change: FeatureFieldChange<AudioToVideoSchema>,
): AudioToVideoValues {
  return applyFeatureFieldChange(values, change);
}

/**
 * Auto has no meaning without a start image, so drop it back to 16:9. A forced shape the
 * toggle no longer shows goes back to the automatic pick, once the audio's length is known.
 */
export function normalizeAudioToVideo(
  values: AudioToVideoValues,
  context: AudioToVideoContext,
): AudioToVideoValues {
  const cell = normalizeA2vCell(values, context);
  return withVisiblePassShape(context.specs, cell, a2vCellSelection(cell, context));
}

function normalizeA2vCell(
  values: AudioToVideoValues,
  context: AudioToVideoContext,
): AudioToVideoValues {
  const model = resolveA2vModel(values.model, context);
  const aspectRatio =
    values.aspectRatio === "auto" && values.startFrame == null
      ? "16:9"
      : values.aspectRatio;
  const resolution = resolveA2vResolution(
    values.resolution,
    context.specs,
    model,
    aspectRatio,
  );
  if (
    model === values.model &&
    resolution === values.resolution &&
    aspectRatio === values.aspectRatio
  ) {
    return values;
  }
  return { ...values, model, resolution, aspectRatio };
}

export function seedAudioToVideoValues(context: AudioToVideoContext): AudioToVideoValues {
  return withDefaultOffering(context.specs, {
    ...AUDIO_TO_VIDEO_DEFAULTS,
    audio: context.seed?.audio ?? null,
    startFrame: context.seed?.startFrame ?? null,
  });
}

export const audioToVideoDefinition: FeatureDefinition<
  AudioToVideoSchema,
  CreateAudioToVideoRequest,
  AudioToVideoContext,
  "audio-to-video"
> = {
  id: "audio-to-video",
  title: "Audio to Video",
  defaults: AUDIO_TO_VIDEO_DEFAULTS,
  initialValues: seedAudioToVideoValues,
  resetValues: seedAudioToVideoValues,
  form: buildAudioToVideoForm,
  applyChange: applyAudioToVideoChange,
  normalize: normalizeAudioToVideo,
  fromGeneration,
  validate: validateAudioToVideo,
  toCreateBody,
  isReady: (values, context) => {
    if (values.audio == null) return false;
    if (values.prompt.trim().length === 0 && values.startFrame == null) {
      return false;
    }
    // Ready when the heard frames fit a budget-kept cell. A prefix of a kept
    // cell stays ready. A dropped ceiling does not. No GenSpace picker-floor
    // probe — a 5s-only machine stays ready.
    return (
      a2vAudioNumFrames(
        context,
        values.model,
        values.resolution,
        values.aspectRatio,
      ) != null
    );
  },
  isUnavailable: (_values, context) =>
    // No downloaded offering with an A2V cell at 540p/24. Settings
    // `local_models` is not a disk catalog.
    !hasAdvertisedA2vCell(context.specs),
};
