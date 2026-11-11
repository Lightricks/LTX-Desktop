import { queueRiskTooltip } from "../../lib/budget-risk.ts";
import {
  CUSTOM_LORA_HOME_FEATURE,
  CUSTOM_LORA_RUN_FALLBACK_LABEL,
} from "../../lib/custom-lora.ts";
import { HOME_FEATURES } from "../../lib/home-features.ts";
import { formatPipelineDisplayName } from "../../lib/video-generation-model-specs.ts";
import { videoGenerationResolutionLabel } from "../../lib/video-resolution.ts";

import { formatA2vDurationBadge } from "./a2vDurationPolicy.ts";
import { formatResolution } from "./formatResolution.ts";
import { IC_LORA_CONTROL_COPY, isIcLoraControlType } from "./icLoraControlType.ts";
import { roundDurationSeconds } from "./roundDuration.ts";
import {
  readGenerationParams,
  type Generation,
} from "./resultsFeedModel.ts";

export { roundDurationSeconds };

type GenerationPresentationSource = Pick<Generation, "feature" | "spec"> & {
  outputs?: Generation["outputs"];
  risk?: Generation["risk"];
};

/**
 * Retake and Extend keep the source picture size unless a smaller one is asked for, and the
 * run caps it. The spec holds the request, not the result, so the label comes from the output.
 */
const OUTPUT_SIZED_FEATURES: ReadonlySet<string> = new Set(["retake", "extend"]);

function outputResolutionLabel(generation: GenerationPresentationSource): string | null {
  if (!OUTPUT_SIZED_FEATURES.has(generation.feature)) return null;
  for (const output of generation.outputs ?? []) {
    if (output.metadata.mediaType === "video") {
      return formatResolution(output.metadata.metadata);
    }
  }
  return null;
}

export function formatDurationBadge(seconds: number): string {
  const rounded = roundDurationSeconds(seconds);
  const label = Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
  return `${label}s`;
}

export type GenerationPresentation = {
  label: string;
  prompt: string;
  badges: string[];
  /** Snapshotted Brave-bet line. Null when the job was not a bet. */
  riskTooltip: string | null;
  /** All input asset IDs for queue / summary UI (frames, audio, etc.). */
  inputAssetIds: string[];
  imageInputAssetIds: string[];
};

function readAssetId(value: unknown): string | null {
  if (!value || typeof value !== "object") return null;
  const assetId = (value as { assetId?: unknown }).assetId;
  return typeof assetId === "string" && assetId.length > 0 ? assetId : null;
}

const INPUT_ASSET_KEY_ORDER: Record<string, readonly string[]> = {
  "image-to-video": ["startFrame", "endFrame"],
  "audio-to-video": ["startFrame", "audio"],
};

const KEYFRAMES_FEATURE = "keyframes";

/** The stills of a Keyframes run, in time order. They sit in a list, not under one key each. */
function keyframeAssetIds(inputRecord: Record<string, unknown>): string[] {
  const keyframes = inputRecord.keyframes;
  if (!Array.isArray(keyframes)) return [];
  const frameOf = (value: unknown): number => {
    const frameIndex = (value as { frameIndex?: unknown } | null)?.frameIndex;
    return typeof frameIndex === "number" ? frameIndex : 0;
  };
  return [...keyframes]
    .sort((a, b) => frameOf(a) - frameOf(b))
    .flatMap((keyframe) => {
      const assetId = readAssetId(keyframe);
      return assetId == null ? [] : [assetId];
    });
}

function readInputAssetIds(generation: GenerationPresentationSource): string[] {
  if (!generation.spec || typeof generation.spec !== "object") return [];
  const inputs = (generation.spec as { inputs?: unknown }).inputs;
  if (!inputs || typeof inputs !== "object" || Array.isArray(inputs)) return [];

  const inputRecord = inputs as Record<string, unknown>;
  if (generation.feature === KEYFRAMES_FEATURE) return keyframeAssetIds(inputRecord);
  const names =
    INPUT_ASSET_KEY_ORDER[generation.feature] ??
    Object.keys(inputRecord);
  return names.flatMap((name) => {
    const assetId = readAssetId(inputRecord[name]);
    return assetId == null ? [] : [assetId];
  });
}

function imageInputAssetIds(generation: GenerationPresentationSource): string[] {
  // Every keyframe is an image.
  if (generation.feature === KEYFRAMES_FEATURE) return readInputAssetIds(generation);
  if (
    generation.feature !== "image-to-video" &&
    generation.feature !== "audio-to-video"
  ) {
    return [];
  }
  const names =
    generation.feature === "image-to-video"
      ? (["startFrame", "endFrame"] as const)
      : (["startFrame"] as const);
  if (!generation.spec || typeof generation.spec !== "object") return [];
  const inputs = (generation.spec as { inputs?: unknown }).inputs;
  if (!inputs || typeof inputs !== "object" || Array.isArray(inputs)) return [];
  const inputRecord = inputs as Record<string, unknown>;
  return names.flatMap((name) => {
    const assetId = readAssetId(inputRecord[name]);
    return assetId == null ? [] : [assetId];
  });
}

/**
 * The strength the run used. Desktop keeps it on `loras[0].scale`. The phone remote withholds
 * the `loras` array (it holds the file path) and sends a top-level `scale`. Undefined when the
 * run has no LoRA, so a plain text-to-video run shows no strength.
 */
function readLoraStrength(params: Record<string, unknown> | null): number | undefined {
  if (!params) return undefined;
  const loras = params.loras;
  const stored =
    Array.isArray(loras) && loras.length > 0
      ? (loras[0] as { scale?: unknown } | null)?.scale
      : params.scale;
  return typeof stored === "number" && Number.isFinite(stored) ? stored : undefined;
}

export function formatLoraStrengthBadge(strength: number): string {
  return `Strength ${Number(strength.toFixed(2))}`;
}

const TEXT_TO_IMAGE_FEATURE = "text-to-image";

/**
 * The name the user gave a saved custom LoRA, as it was when the run started. Desktop keeps it on
 * `loras[0].displayName`. The phone remote withholds `loras` and sends `customLoraName`.
 */
function readCustomLoraName(params: Record<string, unknown> | null): string | null {
  if (!params) return null;
  const loras = params.loras;
  const stored =
    Array.isArray(loras) && loras.length > 0
      ? (loras[0] as { displayName?: unknown } | null)?.displayName
      : params.customLoraName;
  return typeof stored === "string" && stored.trim().length > 0 ? stored : null;
}

function featureLabel(feature: string, params: Record<string, unknown> | null): string {
  // Every saved LoRA shares one feature id. The run is named after its LoRA, as in the sidebar.
  if (feature === CUSTOM_LORA_HOME_FEATURE.id) {
    const name = readCustomLoraName(params);
    return name ?? CUSTOM_LORA_RUN_FALLBACK_LABEL;
  }
  const known = HOME_FEATURES.find((item) => item.id === feature);
  if (known) return known.title;
  if (feature === TEXT_TO_IMAGE_FEATURE) return "Image";
  return "Generation";
}

function imageBadges(params: Record<string, unknown> | null): string[] {
  if (params == null) return [];
  const badges: string[] = [];
  if (typeof params.width === "number" && typeof params.height === "number") {
    badges.push(`${params.width}×${params.height}`);
  }
  if (params.provider === "fal") badges.push("fal");
  else if (params.provider === "local") badges.push("Local");
  return badges;
}

export function generationPresentation(
  generation: GenerationPresentationSource,
): GenerationPresentation {
  const params = readGenerationParams(generation.spec);
  const badges: string[] =
    generation.feature === TEXT_TO_IMAGE_FEATURE ? imageBadges(params) : [];
  const model =
    typeof params?.model === "string"
      ? formatPipelineDisplayName(params.model) ?? params.model
      : null;
  if (model) badges.push(model);
  // Old runs and other recipes carry no `controlType`, or an unknown one: they show no badge.
  if (isIcLoraControlType(params?.controlType)) {
    badges.push(IC_LORA_CONTROL_COPY[params.controlType].label);
  }
  const resolution =
    typeof params?.resolution === "string"
      ? videoGenerationResolutionLabel(params.resolution)
      : outputResolutionLabel(generation);
  if (resolution) badges.push(resolution);
  if (typeof params?.duration === "number") badges.push(formatDurationBadge(params.duration));
  else if (
    typeof params?.numFrames === "number" &&
    typeof params.fps === "number" &&
    params.fps > 0
  ) {
    badges.push(formatA2vDurationBadge(params.numFrames, params.fps));
  }
  if (typeof params?.fps === "number") badges.push(`${params.fps} fps`);
  if (
    params?.aspectRatio === "21:9" ||
    params?.aspectRatio === "16:9" ||
    params?.aspectRatio === "3:2" ||
    params?.aspectRatio === "4:3" ||
    params?.aspectRatio === "1:1" ||
    params?.aspectRatio === "4:5" ||
    params?.aspectRatio === "9:16" ||
    params?.aspectRatio === "auto"
  ) {
    badges.push(params.aspectRatio === "auto" ? "Auto" : params.aspectRatio);
  }
  const strength = readLoraStrength(params);
  if (strength !== undefined) badges.push(formatLoraStrengthBadge(strength));
  if (typeof params?.seed === "number" && Number.isFinite(params.seed)) {
    badges.push(`Seed ${params.seed}`);
  }

  const riskMessage = generation.risk?.message;
  return {
    label: featureLabel(generation.feature, params),
    prompt: typeof params?.prompt === "string" ? params.prompt : "",
    badges,
    riskTooltip:
      typeof riskMessage === "string" && riskMessage.length > 0
        ? queueRiskTooltip(riskMessage, generation.risk?.readout)
        : null,
    inputAssetIds: readInputAssetIds(generation),
    imageInputAssetIds: imageInputAssetIds(generation),
  };
}
