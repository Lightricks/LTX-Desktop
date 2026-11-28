import type { ExploreAsset } from "../../../../lib/explore-contract.ts";

/** What a trim modal needs to open, plus the callback its Done button resolves. */
export type MediaTrimRequest = {
  asset: ExploreAsset;
  durationSeconds: number;
  capSeconds: number;
  /** Replaces the default "not longer than …" line. */
  subtitle?: string;
  /** Rejects when the backend trim fails so the modal can stay open for a retry. */
  onSave: (startSec: number, endSec: number) => Promise<void>;
};

export type MediaImportOutcome =
  | { status: "accepted"; asset: ExploreAsset }
  | { status: "trimming" };

/** Duration of an audio or video asset; null for images. */
export function assetDurationSeconds(asset: ExploreAsset): number | null {
  const { metadata } = asset;
  if (metadata.mediaType !== "audio" && metadata.mediaType !== "video") {
    return null;
  }
  return metadata.metadata.durationMs / 1000;
}

/**
 * A video dropped on an audio field is a valid source, not a mismatch: the
 * Desktop extract step pulls its audio track, and the duration cap applies to
 * that track. Reading the video's own duration keeps an over-long video from
 * skipping the cap.
 */
export const audioDurationSeconds = assetDurationSeconds;

/**
 * One comparison for "too long" so the import gate and form validation agree.
 * `toleranceSeconds` absorbs container rounding (a "1 minute" clip at 60.03s).
 */
export function exceedsDurationCap(
  durationSeconds: number,
  capSeconds: number,
  toleranceSeconds = 0,
): boolean {
  return durationSeconds > capSeconds + toleranceSeconds;
}

/**
 * Shared validate → maybe-trim → accept pipeline for duration-capped audio and
 * video imports, ported from LTX.io's `prepareDurationLimitedMediaImport`.
 * Desktop reads the duration from the ingested asset's metadata rather than
 * re-probing the file in the renderer.
 *
 * A null `trimCapSeconds` means the cap is unknown (model specs have not
 * resolved), so the import is accepted rather than trimmed against a guess.
 */
export async function prepareDurationLimitedMediaImport({
  asset,
  trimCapSeconds,
  toleranceSeconds = 0,
  trim,
  onAccept,
  openTrim,
}: {
  asset: ExploreAsset;
  trimCapSeconds: number | null;
  toleranceSeconds?: number;
  trim: (assetId: string, startSec: number, endSec: number) => Promise<ExploreAsset>;
  onAccept: (asset: ExploreAsset) => void;
  openTrim: (request: MediaTrimRequest) => void;
}): Promise<MediaImportOutcome> {
  const durationSeconds = assetDurationSeconds(asset);

  if (
    trimCapSeconds === null ||
    durationSeconds === null ||
    !exceedsDurationCap(durationSeconds, trimCapSeconds, toleranceSeconds)
  ) {
    onAccept(asset);
    return { status: "accepted", asset };
  }

  openTrim({
    asset,
    durationSeconds,
    capSeconds: trimCapSeconds,
    onSave: async (startSec, endSec) => {
      onAccept(await trim(asset.id, startSec, endSec));
    },
  });
  return { status: "trimming" };
}
