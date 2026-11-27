import { H3_NAMES, H3_VARIANTS, isH3Variant, type H3Variant } from "../../lib/h3-models.ts";
import {
  defaultOfferingId,
  restoreOfferingId,
  type OfferingId,
  type VideoGenerationModelSpecsResponse,
} from "../../lib/video-generation-model-specs.ts";
import type { FieldOption, ValidationIssue } from "../screens/Feature/types.ts";

export type AmdExternalModel = "comfy" | "kandinsky" | H3Variant;
export type VideoPickerModel = OfferingId | AmdExternalModel;
export type AmdPickerContext = {
  preferred: "native" | "comfy";
  desktop: boolean;
  status: "checking" | "offline" | "missing" | "ready" | "downloading" | "error";
  message: string;
  activeNative?: OfferingId;
  native?: Record<OfferingId, "ready" | "missing" | "checking" | "error">;
  kandinskyDownloading?: boolean;
  kandinskyReady: boolean;
  kandinskyMessage: string;
  h3?: Partial<Record<H3Variant, { status: AmdPickerContext["status"]; message: string }>>;
};

export const AMD_MODEL_NAMES = {
  comfy: "LTX 2.5 Fast · ComfyUI INT8",
  kandinsky: "Kandinsky 6 Lite Distill · ComfyUI",
  ...H3_NAMES,
} as const;

export function isExternalModel(model: unknown): model is AmdExternalModel {
  return model === "comfy" || model === "kandinsky" || isH3Variant(model);
}

export function wireModel(model: VideoPickerModel): OfferingId {
  return isExternalModel(model) ? "ltx-2.5-fast" : model;
}

export function selectedEngine(model: VideoPickerModel, amd?: AmdPickerContext) {
  if (isExternalModel(model)) return { amdFastEngine: model };
  return amd && model === "ltx-2.5-fast" ? { amdFastEngine: "native" as const } : {};
}

export function restoreVideoPickerModel(
  params: Record<string, unknown>, specs?: VideoGenerationModelSpecsResponse | null,
): VideoPickerModel {
  if (isExternalModel(params.amdFastEngine)) return params.amdFastEngine;
  return restoreOfferingId(params.model, specs) ?? "ltx-2.5-fast";
}

export function defaultPickerModel(
  specs: VideoGenerationModelSpecsResponse | null | undefined, amd?: AmdPickerContext,
): VideoPickerModel {
  if (amd?.preferred === "comfy" && specs?.comfy_local_model) return "comfy";
  return defaultOfferingId(specs) ?? "ltx-2.5-fast";
}

function modelGrid(specs: VideoGenerationModelSpecsResponse, model: AmdExternalModel) {
  switch (model) {
    case "comfy": return specs.comfy_local_model;
    case "kandinsky": return specs.kandinsky_local_model;
    case "h3-fl2va": return specs.h3_fl2va_local_model;
    case "h3-fl2va-distill": return specs.h3_fl2va_distill_local_model;
    case "h3-ref2va": return specs.h3_ref2va_local_model;
  }
}

export function hasManagedVideoModels(specs?: VideoGenerationModelSpecsResponse): boolean {
  return !!(specs?.comfy_local_model || specs?.kandinsky_local_model ||
    (specs && H3_VARIANTS.some(model => modelGrid(specs, model))));
}

export function pickerGrid(specs: VideoGenerationModelSpecsResponse | null | undefined, model: VideoPickerModel) {
  if (!specs || !isExternalModel(model)) return specs;
  const grid = modelGrid(specs, model);
  return {
    ...specs,
    downloaded_local_models: grid ? [grid] : [],
    active_offering: grid?.model ?? null,
  };
}

function availability(model: AmdExternalModel, amd: AmdPickerContext): string | undefined {
  if (isH3Variant(model)) {
    const variant = amd.h3?.[model];
    if (variant?.status === "ready") return undefined;
    return variant?.message || "Check MiniMax H3 setup on the host PC.";
  }
  if (amd.status === "checking") return "Checking the local video engine…";
  if (amd.status === "offline") return "Set up or start the local video engine in LTX Desktop - AMD 395 MAX.";
  if (model === "kandinsky") {
    return amd.kandinskyReady ? undefined : amd.kandinskyMessage || "Check the Kandinsky nodes and model files in model setup.";
  }
  if (amd.status === "ready") return undefined;
  if (amd.status === "missing") return "The LTX INT8 model files are missing.";
  if (amd.status === "downloading") return "The LTX INT8 model files are downloading.";
  return amd.message || "Check the ComfyUI connection and model files.";
}

function nativeIssue(state: AmdPickerContext["native"], model: OfferingId): string | undefined {
  switch (state?.[model]) {
    case "missing": return "Required model files are missing.";
    case "checking": return "Checking installed model files…";
    case "error": return "Could not check the installed model files.";
    default: return undefined;
  }
}

export function rendererIssue(model: VideoPickerModel, amd?: AmdPickerContext): ValidationIssue | null {
  const message = isExternalModel(model)
    ? (amd ? availability(model, amd) : "This saved model requires a local ComfyUI renderer. Select a model available on this PC to generate again.")
    : nativeIssue(amd?.native, model);
  return message ? { id: "renderer-unavailable", fieldId: "model", message, alwaysRevealed: true } : null;
}

export function pickerOptions(
  nativeOptions: FieldOption<OfferingId>[], specs: VideoGenerationModelSpecsResponse | null | undefined,
  amd: AmdPickerContext, supportsImageInputs = false,
): FieldOption<VideoPickerModel>[] {
  const available = new Set(specs?.downloaded_local_models?.map(item => item.model));
  const native = (["ltx-2.5-fast", "ltx-2.3-fast"] as const).map((model): FieldOption<VideoPickerModel> => {
    const v25 = model === "ltx-2.5-fast";
    const existing = nativeOptions.find(option => option.value === model);
    const fileIssue = nativeIssue(amd.native, model) ??
      (amd.native?.[model] == null && !available.has(model) && !existing ? "Required model files are missing." : undefined);
    const setupKind = v25 ? "ltx-2.5" : "ltx-2.3";
    const needsActivation = amd.desktop && existing && !fileIssue &&
      amd.native?.[model] === "ready" && amd.activeNative != null && amd.activeNative !== model;
    return {
      ...existing, value: model, label: `LTX ${v25 ? "2.5" : "2.3"} Fast · Native`,
      caption: "Native LTX · Desktop PyTorch",
      disabledReason: fileIssue ?? (existing ? undefined : "This model has no compatible settings on this device."),
      ...(fileIssue && { setup: { kind: setupKind,
        label: !amd.desktop ? "Set up on PC" : amd.native?.[model] === "checking" || amd.native?.[model] === "error" ? "Check setup" : "Download & use" } }),
      ...(needsActivation && { setup: { kind: setupKind, label: "Activate" } }),
    };
  });
  function external(model: AmdExternalModel): FieldOption<VideoPickerModel> {
    const isLtx = model === "comfy";
    const disabledReason = availability(model, amd);
    const h3 = isH3Variant(model);
    const downloading = h3 ? amd.h3?.[model]?.status === "downloading" : isLtx ? amd.status === "downloading" : amd.kandinskyDownloading;
    const label = !amd.desktop ? "Set up on PC" : downloading ? "View download" : amd.status === "offline" || amd.status === "checking"
      ? "Set up" : "Get model";
    return {
      value: model, label: AMD_MODEL_NAMES[model],
      caption: h3 ? model === "h3-ref2va" ? "Local reference-image video and audio" : "Local text or keyframe video and audio" : isLtx ? "Local video and audio" : "480p SD · 5 seconds · 24 fps",
      disabledReason,
      ...(disabledReason && { setup: { kind: model === "comfy" ? "comfy-int8" : model, label } }),
    };
  }
  const h3 = H3_VARIANTS.filter(id => specs && modelGrid(specs, id) && (id !== "h3-ref2va" || supportsImageInputs));
  return [
    ...(specs?.comfy_local_model ? [external("comfy")] : []),
    ...native,
    ...(specs?.kandinsky_local_model ? [external("kandinsky")] : []),
    ...h3.map(external),
  ];
}

export function videoModelLabel(engine: unknown): string | undefined {
  return isExternalModel(engine) ? AMD_MODEL_NAMES[engine] : undefined;
}
