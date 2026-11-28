import type { AssetRef } from "../types.ts";

function readAssetRef(value: unknown): AssetRef | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  if (!("assetId" in value) || typeof value.assetId !== "string") {
    return null;
  }
  if (value.assetId.length === 0) return null;
  return { assetId: value.assetId };
}

function readInputSlot(spec: unknown, slot: "startFrame" | "endFrame"): AssetRef | null {
  if (!spec || typeof spec !== "object") return null;
  const inputs = (spec as { inputs?: unknown }).inputs;
  if (!inputs || typeof inputs !== "object" || Array.isArray(inputs)) {
    return null;
  }
  return readAssetRef((inputs as Record<string, unknown>)[slot]);
}

export function readStartFrameFromGeneration(spec: unknown): AssetRef | null {
  return readInputSlot(spec, "startFrame");
}

export function readEndFrameFromGeneration(spec: unknown): AssetRef | null {
  return readInputSlot(spec, "endFrame");
}
