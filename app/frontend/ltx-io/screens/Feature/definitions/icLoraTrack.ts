import type { components } from "../../../../generated/backend-openapi";
import { closestVideoAspectRatio } from "../../../../lib/closest-video-aspect-ratio.ts";
import type { IcLoraTrackRecipe } from "../../../../lib/ic-lora-recipes.ts";
import {
  type OfferingId, VIDEO_GENERATION_ASPECT_RATIOS, withDefaultOffering,
  type VideoGenerationAspectRatio, type VideoGenerationDuration,
  type VideoGenerationFps, type VideoGenerationResolution,
} from "../../../../lib/video-generation-model-specs.ts";
import motionPathShared from "../../../../../shared/motion-path.json" with { type: "json" };
import { clampMotionPathHolds, MOTION_PATH_MAX_TRACKS, type MotionPath } from "../../../lib/motionPathSampler.ts";
import { readWireMotionPath, toEditorPath, toWireMotionPath } from "../../../lib/motionPathWire.ts";
import { readGenerationParams } from "../../../lib/resultsFeedModel.ts";
import { isLocalVideoSelectionReady } from "../../../lib/videoFieldPolicy.ts";
import { applyVideoSettingsChange } from "../../../lib/videoSettingsControls.ts";
import { buildMemoryMeterModel } from "../../../../lib/memory-meter.ts";
import {
  type AssetRef, type FeatureDefinition, type FeatureFormModel, type FeatureValues, type ValidationIssue,
} from "../types.ts";
import { icLoraVariantId } from "./icLoraNormalize.ts";
import {
  type IcLoraAudioMode, isIcLoraAudioMode, isSynthesizedIcLoraResolution,
  isSynthesizedIcLoraUnavailable, normalizeSynthesizedIcLoraValues, readSynthesizedCommonParams,
  type SynthesizedIcLoraCatalogSettings, type SynthesizedIcLoraContext,
  synthesizedIcLoraVideoOptionFields,
} from "./icLoraSynthesizedCommon.ts";
import { negativePromptField, swapDemoNegative } from "./negativePrompt.ts";
import { readAssetRef } from "./videoFeature.ts";

type CreateIcLoraTrackRequest = components["schemas"]["CreateIcLoraTrackRequest"];

export type IcLoraTrackSchema = {
  /** The start frame. The model reads it as the picture the path moves over. */
  startFrame: { kind: "image-asset"; value: AssetRef | null };
  /** The tracks. The wire body holds them as `inputs.paths`. */
  paths: { kind: "motion-path"; value: MotionPath[] };
  prompt: { kind: "textarea" };
  negativePrompt: { kind: "textarea" };
  model: { kind: "options"; value: OfferingId };
  variant: { kind: "options"; value: string };
  /** Never shown. The backend takes the closest ratio from the image, so the form clamps as for Auto. */
  aspectRatio: { kind: "options"; value: "auto" };
  resolution: { kind: "options"; value: VideoGenerationResolution };
  duration: { kind: "options"; value: VideoGenerationDuration };
  fps: { kind: "options"; value: VideoGenerationFps };
  audioMode: { kind: "options"; value: IcLoraAudioMode };
  loraStrength: { kind: "slider"; value: number };
};

export type IcLoraTrackValues = FeatureValues<IcLoraTrackSchema>;

export type IcLoraTrackContext = SynthesizedIcLoraContext & {
  /** The packaged start frame of the example. It fills a new form and Reset. */
  seed?: AssetRef;
  /** The start frame dimensions, so the form budget prices the closest ratio as create does. */
  startFrameSize?: { width: number; height: number } | null;
};

export type IcLoraTrackCatalogSettings = SynthesizedIcLoraCatalogSettings;

/** The path of the ltx.io example (the conductor frog). The shared file holds it for the backend tests. */
const EXAMPLE_PATH =
  readWireMotionPath(motionPathShared.examplePath) ?? invalidExamplePath();

function invalidExamplePath(): never {
  throw new Error("shared/motion-path.json holds an invalid example path");
}

function defaultsFor(
  recipe: IcLoraTrackRecipe,
  settings?: IcLoraTrackCatalogSettings | null,
): IcLoraTrackValues {
  const audio = settings?.audio_mode;
  return {
    startFrame: null,
    paths: [],
    prompt: recipe.demoPrompt,
    negativePrompt: recipe.demoNegativePrompt || settings?.default_negative_prompt || "",
    model: "ltx-2.5-fast",
    variant: "",
    aspectRatio: "auto",
    resolution: "720p",
    duration: recipe.homeDefaults.seconds,
    fps: recipe.homeDefaults.fps,
    audioMode: isIcLoraAudioMode(audio) ? audio : "generated",
    loraStrength: recipe.homeDefaults.loraStrength,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** The start frame and the tracks of a saved spec. */
export function readTrack(spec: unknown): { startFrame: AssetRef; paths: MotionPath[] } | null {
  if (!isRecord(spec) || !isRecord(spec.inputs)) return null;
  const startFrame = readAssetRef(spec.inputs.image);
  const wirePaths = Array.isArray(spec.inputs.paths) ? spec.inputs.paths.map(readWireMotionPath) : [];
  const paths = wirePaths.flatMap((wirePath) => (wirePath ? [toEditorPath(wirePath)] : []));
  const isValid =
    paths.length > 0 && paths.length === wirePaths.length && paths.length <= MOTION_PATH_MAX_TRACKS;
  return startFrame && isValid ? { startFrame, paths } : null;
}

/** One to three tracks, and every track has two points. */
function hasDrawnPaths(paths: MotionPath[]): boolean {
  return (
    paths.length > 0 &&
    paths.length <= MOTION_PATH_MAX_TRACKS &&
    paths.every((path) => path.points.length >= 2)
  );
}

function trackAspectRatio(context: IcLoraTrackContext): VideoGenerationAspectRatio {
  const size = context.startFrameSize;
  return size != null && size.height > 0
    ? (closestVideoAspectRatio(size.width, size.height, VIDEO_GENERATION_ASPECT_RATIOS) ?? "16:9")
    : "16:9";
}

/**
 * The tracks hold positions on the image they were drawn on, so a new image clears them. When
 * the duration shortens, the pauses of each track clamp so they still leave time for its dot.
 */
function reconcilePathOnEdit(
  previous: IcLoraTrackValues,
  next: IcLoraTrackValues,
): IcLoraTrackValues {
  if (previous.startFrame?.assetId !== next.startFrame?.assetId) {
    return { ...next, paths: [] };
  }
  let changed = false;
  const paths = next.paths.map((path) => {
    const holds = clampMotionPathHolds({ durationSec: next.duration, ...path });
    if (holds.startHoldSec === path.startHoldSec && holds.endHoldSec === path.endHoldSec) {
      return path;
    }
    changed = true;
    return { ...path, ...holds };
  });
  return changed ? { ...next, paths } : next;
}

export function createIcLoraTrackDefinition(
  recipe: IcLoraTrackRecipe,
  settings?: IcLoraTrackCatalogSettings | null,
): FeatureDefinition<IcLoraTrackSchema, CreateIcLoraTrackRequest, IcLoraTrackContext, IcLoraTrackRecipe["id"]> {
  const defaults = defaultsFor(recipe, settings);

  function fromGeneration(spec: unknown, context: IcLoraTrackContext): IcLoraTrackValues | null {
    const params = readGenerationParams(spec);
    const track = readTrack(spec);
    if (!params || !track) return null;
    return {
      startFrame: track.startFrame,
      paths: track.paths,
      aspectRatio: "auto",
      ...readSynthesizedCommonParams(params, context, defaults),
    };
  }

  function normalize(values: IcLoraTrackValues, context: IcLoraTrackContext): IcLoraTrackValues {
    const normalized = normalizeSynthesizedIcLoraValues<IcLoraTrackSchema>(
      values,
      context,
      defaults,
      trackAspectRatio(context),
    );
    return reconcilePathOnEdit(values, normalized);
  }

  function validate(values: IcLoraTrackValues): ValidationIssue[] {
    const issues: ValidationIssue[] = [];
    if (values.startFrame == null) {
      issues.push({
        id: "image-required",
        fieldId: "path",
        message: "Add a start frame.",
        alwaysRevealed: true,
      });
    } else if (!hasDrawnPaths(values.paths)) {
      issues.push({
        id: "path-required",
        fieldId: "path",
        message: "Create a motion path",
        alwaysRevealed: true,
      });
    }
    if (values.prompt.trim() === "") {
      issues.push({
        id: "prompt-required",
        fieldId: "prompt",
        message: "Prompt is required.",
        alwaysRevealed: true,
      });
    }
    return issues;
  }

  function toCreateBody(
    values: IcLoraTrackValues,
    context?: IcLoraTrackContext,
  ): CreateIcLoraTrackRequest {
    const { startFrame, paths } = values;
    if (startFrame == null || !hasDrawnPaths(paths)) {
      throw new Error("A start frame and paths of two points each are required.");
    }
    if (!isSynthesizedIcLoraResolution(values.resolution)) {
      throw new Error("A valid resolution is required.");
    }
    const variantId = icLoraVariantId(values, context);
    return {
      contract_version: 1,
      inputs: {
        image: { assetId: startFrame.assetId },
        paths: paths.map(toWireMotionPath),
      },
      params: {
        prompt: values.prompt.trim(),
        negativePrompt: values.negativePrompt.trim(),
        model: values.model,
        resolution: values.resolution,
        duration: values.duration,
        fps: values.fps,
        audioMode: values.audioMode,
        scale: values.loraStrength,
        ...(variantId ? { variantId } : {}),
        promptProvenance: "typed",
      },
    };
  }

  function form(
    values: IcLoraTrackValues,
    context: IcLoraTrackContext,
  ): FeatureFormModel<IcLoraTrackSchema> {
    const { duration, tail } = synthesizedIcLoraVideoOptionFields<IcLoraTrackSchema>(
      values,
      context,
      trackAspectRatio(context),
    );
    return {
      fields: [
        {
          kind: "motion-path",
          id: "path",
          label: "Motion path",
          dataKey: "paths",
          imageKey: "startFrame",
          durationKey: "duration",
          durationOptions: duration.options,
        },
        {
          kind: "textarea",
          id: "prompt",
          label: "Prompt",
          dataKey: "prompt",
          placeholder: recipe.placeholder,
        },
        negativePromptField<IcLoraTrackSchema>(),
        ...tail,
      ],
      memory: buildMemoryMeterModel(
        context.specs,
        { ...values, aspectRatio: trackAspectRatio(context) },
        { fixFields: ["duration", "resolution"] },
      ),
    };
  }

  function seedValues(context: IcLoraTrackContext): IcLoraTrackValues {
    return {
      ...withDefaultOffering(context.specs, defaults),
      startFrame: context.seed ?? null,
      paths: context.seed ? [toEditorPath(EXAMPLE_PATH)] : [],
    };
  }

  return {
    id: recipe.id,
    title: recipe.title,
    defaults,
    initialValues: seedValues,
    resetValues: seedValues,
    form,
    applyChange: (values, change, context) => {
      const applied = applyVideoSettingsChange<IcLoraTrackSchema>(
        values,
        change,
        context.specs,
        withDefaultOffering(context.specs, defaults),
      );
      return swapDemoNegative(values, normalize(reconcilePathOnEdit(values, applied), context), {
        demoNegative: recipe.demoNegativePrompt,
        catalogDefault: settings?.default_negative_prompt,
        changed: (before, after) =>
          (change.dataKey === "startFrame" || change.dataKey === "prompt") &&
          (after.startFrame?.assetId !== before.startFrame?.assetId ||
            after.prompt !== before.prompt),
      });
    },
    normalize,
    fromGeneration,
    validate,
    toCreateBody,
    // Generate stays off until a start frame, a path, and a prompt exist.
    isReady: (values, context) =>
      values.startFrame != null &&
      hasDrawnPaths(values.paths) &&
      values.prompt.trim() !== "" &&
      isLocalVideoSelectionReady({ ...values, aspectRatio: trackAspectRatio(context) }, context.specs),
    isUnavailable: (values, context) =>
      isSynthesizedIcLoraUnavailable({ ...values, aspectRatio: trackAspectRatio(context) }, context),
  };
}
