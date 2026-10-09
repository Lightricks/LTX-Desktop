import type { AssetsMediaKind } from "./assetsSearchParams";

export type AssetPickerDrop =
  | { kind: "asset"; id: string }
  | { kind: "file"; file: File }
  | { kind: "ignore" };

/**
 * A drag onto the picker is either an in-app asset (`asset` JSON from
 * GenSpace / the editor) or a file from outside the app. An asset whose
 * kind we can read is used only when it matches; anything else is ignored
 * so a photo drop does not replace a video source.
 */
export function resolveAssetPickerDrop({
  assetPayload,
  file,
  mediaKind,
  isEligibleFile,
}: {
  assetPayload: string;
  file: File | null;
  mediaKind: AssetsMediaKind;
  isEligibleFile: (file: File) => boolean;
}): AssetPickerDrop {
  const dropped = parseDroppedAsset(assetPayload);
  if (dropped != null) {
    return dropped.mediaKind === mediaKind
      ? { kind: "asset", id: dropped.id }
      : { kind: "ignore" };
  }
  if (file != null && isEligibleFile(file)) {
    return { kind: "file", file };
  }
  return { kind: "ignore" };
}

function parseDroppedAsset(payload: string): { id: string; mediaKind: string } | null {
  if (payload === "") return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(payload);
  } catch {
    return null;
  }
  if (typeof parsed !== "object" || parsed === null) return null;
  const record = parsed as { id?: unknown; media_kind?: unknown; type?: unknown };
  if (typeof record.id !== "string" || record.id.length === 0) return null;
  const mediaKind = record.media_kind ?? record.type;
  if (typeof mediaKind !== "string" || mediaKind.length === 0) return null;
  return { id: record.id, mediaKind };
}
