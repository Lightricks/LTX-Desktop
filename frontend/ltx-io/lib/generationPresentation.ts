import { HOME_FEATURES } from "../../lib/home-features.ts";
import { formatPipelineDisplayName } from "../../lib/video-generation-model-specs.ts";

import { formatA2vDurationBadge } from "./a2vDurationPolicy.ts";
import { roundDurationSeconds } from "./roundDuration.ts";
import {
  readGenerationParams,
  type Generation,
} from "./resultsFeedModel.ts";

export { roundDurationSeconds };

type GenerationPresentationSource = Pick<Generation, "feature" | "spec">;

export function formatDurationBadge(seconds: number): string {
  const rounded = roundDurationSeconds(seconds);
  const label = Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
  return `${label}s`;
}

export type GenerationPresentation = {
  label: string;
  prompt: string;
  badges: string[];
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

function readInputAssetIds(generation: GenerationPresentationSource): string[] {
  if (!generation.spec || typeof generation.spec !== "object") return [];
  const inputs = (generation.spec as { inputs?: unknown }).inputs;
  if (!inputs || typeof inputs !== "object" || Array.isArray(inputs)) return [];

  const inputRecord = inputs as Record<string, unknown>;
  const names =
    INPUT_ASSET_KEY_ORDER[generation.feature] ??
    Object.keys(inputRecord);
  return names.flatMap((name) => {
    const assetId = readAssetId(inputRecord[name]);
    return assetId == null ? [] : [assetId];
  });
}

function imageInputAssetIds(generation: GenerationPresentationSource): string[] {
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

function featureLabel(feature: string): string {
  const known = HOME_FEATURES.find((item) => item.id === feature);
  if (known) return known.title;
  return "Generation";
}

export function generationPresentation(
  generation: GenerationPresentationSource,
): GenerationPresentation {
  const params = readGenerationParams(generation.spec);
  const badges: string[] = [];
  const model =
    typeof params?.model === "string"
      ? formatPipelineDisplayName(params.model) ?? params.model
      : null;
  if (model) badges.push(model);
  if (typeof params?.resolution === "string") badges.push(params.resolution);
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
  if (typeof params?.seed === "number" && Number.isFinite(params.seed)) {
    badges.push(`Seed ${params.seed}`);
  }

  return {
    label: featureLabel(generation.feature),
    prompt: typeof params?.prompt === "string" ? params.prompt : "",
    badges,
    inputAssetIds: readInputAssetIds(generation),
    imageInputAssetIds: imageInputAssetIds(generation),
  };
}
