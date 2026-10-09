import {
  isOfferingId,
  restoreOfferingId,
  type DownloadedLocalVideoGenerationModelSpecItem,
  type OfferingId,
} from "../../../../lib/video-generation-model-specs.ts";
import { lowPerformanceResolutionWarning } from "../../../lib/lowPerformanceWarning.ts";
import { readGenerationParams } from "../../../lib/resultsFeedModel.ts";
import type { VideoFeatureContext } from "../../../lib/videoFieldPolicy.ts";
import type { AssetRef, FieldOption, ValidationIssue } from "../types.ts";
import {
  CELL_SHORT_EDGE,
  IC_LORA_RESOLUTIONS,
  icLoraResolutionOptions,
  type IcLoraRecipeResolution,
} from "./icLoraResolution.ts";

export type VideoInputContext = VideoFeatureContext & {
  seed: AssetRef | null;
  videoDurationSeconds: number | null;
  videoDurationPending: boolean;
  videoWidth: number | null;
  videoHeight: number | null;
  videoFps: number | null;
};

/** "original" keeps the source size. Any other key is an IC-LoRA cell below it. */
export type EditResolutionKey = "original" | IcLoraRecipeResolution;

function resolutionKeyFromSize(width: number, height: number): IcLoraRecipeResolution {
  const shortEdge = Math.min(width, height);
  return IC_LORA_RESOLUTIONS.reduce((best, cell) =>
    Math.abs(CELL_SHORT_EDGE[cell] - shortEdge) < Math.abs(CELL_SHORT_EDGE[best] - shortEdge)
      ? cell
      : best,
  );
}

export function editResolutionKeyFromParams(
  params: Record<string, unknown>,
): EditResolutionKey {
  const resolution = params.resolution;
  if (!resolution || typeof resolution !== "object") return "original";
  const record = resolution as { width?: unknown; height?: unknown };
  if (typeof record.width !== "number" || typeof record.height !== "number") {
    return "original";
  }
  return resolutionKeyFromSize(record.width, record.height);
}

/**
 * The IC-LoRA cells below the clip, then Original. Retake and extend never upscale.
 * Original runs at the source size, capped at the 1080p local cell, so its label
 * names that size. Empty when the size is unknown or no cell is below the clip,
 * so the control stays hidden.
 */
export function editResolutionFieldOptions(
  width: number | null,
  height: number | null,
  lowPerformanceMachine?: boolean,
): FieldOption<EditResolutionKey>[] {
  if (width == null || height == null) return [];
  const sourceEdge = Math.min(Math.min(width, height), CELL_SHORT_EDGE["1080p"]);
  const lower = icLoraResolutionOptions({ videoWidth: width, videoHeight: height, specs: null })
    .filter((option) => CELL_SHORT_EDGE[option.value] < sourceEdge)
    .map((option): [EditResolutionKey, string, number] => [
      option.value,
      option.value,
      CELL_SHORT_EDGE[option.value],
    ]);
  if (lower.length === 0) return [];
  const original: [EditResolutionKey, string, number] = [
    "original",
    `${sourceEdge}p (Original)`,
    sourceEdge,
  ];
  return [...lower, original].map(([value, label, edge]) => {
    const warning = lowPerformanceResolutionWarning(edge, lowPerformanceMachine);
    return { value, label, ...(warning ? { warning } : {}) };
  });
}

export function normalizeEditResolution(
  current: EditResolutionKey,
  width: number | null,
  height: number | null,
): EditResolutionKey {
  if (width == null || height == null) return current;
  const offered = editResolutionFieldOptions(width, height).some((option) => option.value === current);
  return offered ? current : "original";
}

export function editResolutionSize(
  key: EditResolutionKey,
  width: number | null,
  height: number | null,
): { width: number; height: number } | undefined {
  if (key === "original" || width == null || height == null) return undefined;
  const short = CELL_SHORT_EDGE[key];
  if (short >= Math.min(width, height)) return undefined;
  const long = Math.round((Math.max(width, height) * short) / Math.min(width, height));
  return width >= height ? { width: long, height: short } : { width: short, height: long };
}

export function readAssetRef(value: unknown): AssetRef | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  if (!("assetId" in value) || typeof value.assetId !== "string") return null;
  if (value.assetId.length === 0) return null;
  return { assetId: value.assetId };
}

export function readGenerationInputs(
  spec: unknown,
  key: string,
): Record<string, AssetRef | null> | null {
  if (!spec || typeof spec !== "object") return null;
  const inputs = (spec as { inputs?: unknown }).inputs;
  if (!inputs || typeof inputs !== "object" || Array.isArray(inputs)) {
    return null;
  }
  const record = inputs as Record<string, unknown>;
  if (!(key in record)) return null;
  return { [key]: readAssetRef(record[key]) };
}

export function readVideoGenerationInput(spec: unknown): { video: AssetRef } | null {
  const inputs = readGenerationInputs(spec, "video");
  const video = inputs?.video ?? null;
  if (!video) return null;
  return { video };
}

export type VideoCapability = "retake" | "extend" | "ic_lora";

function offeringAdvertises(
  spec: DownloadedLocalVideoGenerationModelSpecItem["spec"],
  capability: VideoCapability,
): boolean {
  return spec.capabilities?.[capability] === true;
}

export function capabilityModelOptions(
  specs: VideoFeatureContext["specs"],
  capability: VideoCapability,
): FieldOption<OfferingId>[] {
  return (specs?.downloaded_local_models ?? [])
    .filter(
      (item) => isOfferingId(item.model) && offeringAdvertises(item.spec, capability),
    )
    .map((item) => ({
      value: item.model,
      label: item.spec.display_name,
    }));
}

export function hasAdvertisedCapability(
  specs: VideoFeatureContext["specs"],
  capability: VideoCapability,
): boolean {
  return capabilityModelOptions(specs, capability).length > 0;
}

export function resolveCapableModel(
  model: unknown,
  specs: VideoFeatureContext["specs"],
  capability: VideoCapability,
  fallback: OfferingId,
): OfferingId {
  const options = capabilityModelOptions(specs, capability);
  if (isOfferingId(model) && options.some((option) => option.value === model)) {
    return model;
  }
  return options[0]?.value ?? fallback;
}

export function restoreModel(
  params: Record<string, unknown> | null,
  specs: VideoFeatureContext["specs"],
  fallback: OfferingId,
): OfferingId {
  if (params == null) return fallback;
  return restoreOfferingId(params.model, specs) ?? fallback;
}

export function videoLengthUnknownIssue(): ValidationIssue {
  return {
    id: "video-duration-unknown",
    fieldId: "video",
    message: "We couldn't read this video's length. Replace it to continue.",
    alwaysRevealed: true,
  };
}

export function videoRequiredIssue(): ValidationIssue {
  return {
    id: "video-required",
    fieldId: "video",
    message: "Add a video.",
  };
}

export function readPrompt(spec: unknown): string {
  const params = readGenerationParams(spec);
  return typeof params?.prompt === "string" ? params.prompt : "";
}
