import type { components } from "../../../../generated/backend-openapi";
import {
  isOfferingId,
  isVideoGenerationAspectRatio,
  restoreOfferingId,
  untrainedAspectRatioWarning,
  orderAspectRatiosForPicker,
  withDefaultOffering,
  type OfferingId,
} from "../../../../lib/video-generation-model-specs.ts";
import {
  A2V_FPS,
  A2V_MIN_AUDIO_SECONDS,
  A2V_RESOLUTION,
  a2vAspectRatios,
  a2vDurationsAt540p24FromSpec,
  a2vEffectiveAudioSeconds,
  a2vLongestCellSeconds,
  a2vNumFramesForAudio,
  a2vResolutionsForOffering,
  a2vTrimCapSeconds,
  advertisedA2vDurations,
  hasAdvertisedA2vCell,
  isA2vResolution,
  type A2vResolution,
} from "../../../lib/a2vDurationPolicy.ts";
import { readGenerationParams } from "../../../lib/resultsFeedModel.ts";
import type {
  VideoFeatureAspectRatio,
  VideoFeatureContext,
} from "../../../lib/videoFieldPolicy.ts";
import { VIDEO_SETTING_RESOLUTIONS } from "../../../lib/videoSettingsControls.ts";
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
import { readAssetRef } from "./videoFeature.ts";

type CreateAudioToVideoRequest = components["schemas"]["CreateAudioToVideoRequest"];

export type AudioToVideoAspectRatio = VideoFeatureAspectRatio;

export type AudioToVideoSchema = {
  startFrame: { kind: "image-asset"; value: AssetRef | null };
  prompt: { kind: "textarea" };
  audio: { kind: "audio-asset"; value: AssetRef | null };
  model: { kind: "options"; value: OfferingId };
  resolution: { kind: "options"; value: A2vResolution };
  aspectRatio: { kind: "options"; value: AudioToVideoAspectRatio };
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
};

const AUTO_ASPECT_OPTION: FieldOption<AudioToVideoAspectRatio> = {
  value: "auto",
  label: "Auto",
};

function explicitAspectOptions(
  context: AudioToVideoContext,
  offering: OfferingId,
  resolution: A2vResolution,
): FieldOption<AudioToVideoAspectRatio>[] {
  return orderAspectRatiosForPicker(
    a2vAspectRatios(context.specs, offering, resolution),
  ).map((value) => {
    const warning = untrainedAspectRatioWarning(value);
    return {
      value,
      label: value,
      ...(warning ? { warning } : {}),
    };
  });
}

/** Same copy as studio's A2V lipsync placeholder — empty-state form seed. */
export const AUDIO_TO_VIDEO_EXAMPLE_PROMPT =
  "The character speaks while moving through the scene";

/**
 * Empty state: example prompt plus Auto aspect, which matches the packaged
 * example start image. Normalize rewrites Auto to 16:9 when no image is set.
 */
export const AUDIO_TO_VIDEO_DEFAULTS: AudioToVideoValues = {
  startFrame: null,
  prompt: AUDIO_TO_VIDEO_EXAMPLE_PROMPT,
  audio: null,
  model: "ltx-2.5-fast",
  resolution: A2V_RESOLUTION,
  aspectRatio: "auto",
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
  const startFrame = readAssetRef(
    (inputs as { startFrame?: unknown }).startFrame,
  );
  return { audio, startFrame };
}

export function a2vModelOptions(
  specs: AudioToVideoContext["specs"],
): FieldOption<OfferingId>[] {
  return (specs?.downloaded_local_models ?? [])
    .filter(
      (item) =>
        isOfferingId(item.model) &&
        a2vDurationsAt540p24FromSpec(item.spec).length > 0,
    )
    .map((item) => ({
      value: item.model,
      label: item.spec.display_name,
    }));
}

function a2vResolutionOptions(
  specs: AudioToVideoContext["specs"],
  offering: OfferingId,
): FieldOption<A2vResolution>[] {
  return a2vResolutionsForOffering(specs, offering).map((resolution) => ({
    value: resolution,
    label: resolution,
  }));
}

function resolveA2vModel(
  model: unknown,
  context: AudioToVideoContext,
): OfferingId {
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
): A2vResolution {
  const options = a2vResolutionsForOffering(specs, offering);
  if (isA2vResolution(resolution) && options.includes(resolution)) {
    return resolution;
  }
  return options[0] ?? A2V_RESOLUTION;
}

/**
 * Client-side A2V frame estimate for the generate gate. Null when the audio
 * has not loaded, is under 2s, exceeds every advertised cell, or the machine
 * advertises no A2V cells. A clip past this resolution but inside a longer
 * cell is estimated from that cell's max (the opening prefix). Advertised
 * cells are the envelope, not snap targets; the server derives and persists
 * the authoritative `numFrames`.
 */
export function a2vAudioNumFrames(
  context: AudioToVideoContext,
  offering?: OfferingId | null,
  resolution: A2vResolution = A2V_RESOLUTION,
): number | null {
  if (context.audioDurationSeconds == null) return null;
  const advertised = advertisedA2vDurations(context.specs, offering, resolution);
  const effective = a2vEffectiveAudioSeconds(
    context.audioDurationSeconds,
    advertised,
    a2vLongestCellSeconds(context.specs, offering),
  );
  if (effective == null) return null;
  return a2vNumFramesForAudio(effective, advertised, A2V_FPS);
}

export function fromGeneration(
  spec: unknown,
  context: AudioToVideoContext = {
    specs: null,
    seed: null,
    audioDurationSeconds: null,
    audioDurationPending: false,
  },
): AudioToVideoValues | null {
  const params = readGenerationParams(spec);
  const inputs = readGenerationInputs(spec);
  if (!params || !inputs) return null;
  const model =
    restoreOfferingId(params.model, context.specs) ??
    AUDIO_TO_VIDEO_DEFAULTS.model;

  return {
    startFrame: inputs.startFrame,
    prompt: typeof params.prompt === "string" ? params.prompt : "",
    audio: inputs.audio,
    model,
    resolution: resolveA2vResolution(params.resolution, context.specs, model),
    aspectRatio: isAspectRatio(params.aspectRatio)
      ? params.aspectRatio
      : AUDIO_TO_VIDEO_DEFAULTS.aspectRatio,
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
  const advertised = advertisedA2vDurations(
    context.specs,
    values.model,
    values.resolution,
  );
  const effective = a2vEffectiveAudioSeconds(
    context.audioDurationSeconds,
    advertised,
    a2vLongestCellSeconds(context.specs, values.model),
  );
  // Shorter than the clip means this resolution keeps the opening prefix.
  // The same helper feeds the Generate gate, so the notice cannot drift from it.
  if (effective != null && effective < context.audioDurationSeconds) {
    return {
      id: "audio-longer-than-resolution",
      fieldId: "audio",
      message: `At ${values.resolution}, the maximum is ${effective} seconds. We'll use only the first ${effective}s, or you can choose a lower resolution.`,
      alwaysRevealed: true,
      blocksGenerate: false,
    };
  }
  if (effective != null) return null;
  const cap =
    advertised.length === 0
      ? a2vTrimCapSeconds(context.specs, values.model)
      : advertised[advertised.length - 1];
  // No cap means the machine advertises no A2V cell at all, which the
  // feature-unavailable message already explains.
  if (cap == null) return null;
  return {
    id: "audio-longer-than-cap",
    fieldId: "audio",
    message: `This audio is longer than ${cap}s at ${values.resolution}. Choose a lower resolution or replace it with a shorter clip.`,
    alwaysRevealed: true,
  };
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
  return issues;
}

export function toCreateBody(
  values: AudioToVideoValues,
  context: AudioToVideoContext,
): CreateAudioToVideoRequest {
  if (values.audio == null) {
    throw new Error("Audio is required.");
  }
  if (a2vAudioNumFrames(context, values.model, values.resolution) == null) {
    throw new Error("This audio is longer than this machine can generate.");
  }
  return {
    contract_version: 1,
    inputs: {
      audio: { assetId: values.audio.assetId },
      startFrame:
        values.startFrame == null
          ? null
          : { assetId: values.startFrame.assetId },
    },
    params: {
      prompt: values.prompt.trim(),
      model: values.model,
      aspectRatio: values.aspectRatio,
      resolution: values.resolution,
      fps: A2V_FPS,
      cameraMotion: "none",
      negativePrompt: "",
      promptProvenance: "typed",
    },
  };
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
        options: a2vResolutionOptions(context.specs, values.model),
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
          ...explicitAspectOptions(context, values.model, values.resolution),
        ],
      },
    ],
  };
}

export function applyAudioToVideoChange(
  values: AudioToVideoValues,
  change: FeatureFieldChange<AudioToVideoSchema>,
): AudioToVideoValues {
  return applyFeatureFieldChange(values, change);
}

/** Auto has no meaning without a start image, so drop it back to 16:9. */
export function normalizeAudioToVideo(
  values: AudioToVideoValues,
  context: AudioToVideoContext,
): AudioToVideoValues {
  const model = resolveA2vModel(values.model, context);
  const resolution = resolveA2vResolution(
    values.resolution,
    context.specs,
    model,
  );
  const next =
    model === values.model && resolution === values.resolution
      ? values
      : { ...values, model, resolution };
  if (next.aspectRatio === "auto" && next.startFrame == null) {
    return { ...next, aspectRatio: "16:9" };
  }
  return next;
}

export function seedAudioToVideoValues(
  context: AudioToVideoContext,
): AudioToVideoValues {
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
    // A2V readiness is the envelope check: >=2s and inside the longest cell.
    // A clip past this resolution but inside a longer cell stays ready —
    // generation uses that cell's opening prefix. No GenSpace picker-floor
    // probe — a 5s-only machine stays ready.
    return a2vAudioNumFrames(context, values.model, values.resolution) != null;
  },
  isUnavailable: (_values, context) =>
    // No downloaded offering with an A2V cell at 540p/24. Settings
    // `local_models` is not a disk catalog.
    !hasAdvertisedA2vCell(context.specs),
};
