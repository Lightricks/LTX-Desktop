import {
  defaultOfferingId,
  isVideoGenerationAspectRatio,
  type OfferingId,
  type VideoGenerationAspectRatio,
  type VideoGenerationModelSpecItem,
  type VideoGenerationModelSpecsResponse,
} from "../../lib/video-generation-model-specs.ts";

export const A2V_RESOLUTION = "540p" as const;
export const A2V_FPS = 24 as const;
export const A2V_RESOLUTIONS = ["270p", "360p", "540p", "720p", "1080p"] as const;
export type A2vResolution = (typeof A2V_RESOLUTIONS)[number];
/** API-equivalent A2V minimum: gateway rejects under 2s. */
export const A2V_MIN_AUDIO_SECONDS = 2 as const;
/**
 * Shortest selection the A2V trim modal accepts. A trimmed clip becomes the
 * generation input, so the floor is the 2s audio minimum — GenSpace's 6s
 * picker floor must not apply, or a 5s-only envelope has no usable range.
 */
export const A2V_TRIM_MIN_SECONDS = A2V_MIN_AUDIO_SECONDS;
/** Probe slack matching ltxv-api `maxDurationTolerance = 0.1`. */
export const A2V_CAP_TOLERANCE_SECONDS = 0.1 as const;

type A2vSpec = VideoGenerationModelSpecItem["spec"];

export function isA2vResolution(value: unknown): value is A2vResolution {
  return typeof value === "string" && (A2V_RESOLUTIONS as readonly string[]).includes(value);
}

function a2vDurationsFromSpec(spec: A2vSpec, resolution: A2vResolution): number[] {
  const map = spec.a2v_supported_resolutions_durations ?? {};
  const list = map[resolution]?.fps_to_durations?.[String(A2V_FPS)] ?? [];
  return [...list].sort((a, b) => a - b);
}

export function a2vDurationsAt540p24FromSpec(spec: A2vSpec): number[] {
  return a2vDurationsFromSpec(spec, A2V_RESOLUTION);
}

/**
 * Hardware envelope at the fixed A2V cell (540p/24). Downloaded offerings
 * only — Settings `local_models` is the active pipeline row, not a disk
 * catalog. A missing selected offering is no cell, not a Settings fallback.
 */
export function advertisedA2vDurationsAt540p24(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  offering?: OfferingId | null,
): number[] {
  const downloaded = specs?.downloaded_local_models ?? [];
  if (offering != null) {
    const selected = downloaded.find((item) => item.model === offering);
    return selected ? a2vDurationsAt540p24FromSpec(selected.spec) : [];
  }
  const preferred = defaultOfferingId(specs);
  const selected =
    downloaded.find((item) => item.model === preferred) ?? downloaded[0];
  return selected ? a2vDurationsAt540p24FromSpec(selected.spec) : [];
}

function specForOffering(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  offering?: OfferingId | null,
): A2vSpec | null {
  const downloaded = specs?.downloaded_local_models ?? [];
  if (offering != null) {
    return downloaded.find((item) => item.model === offering)?.spec ?? null;
  }
  const preferred = defaultOfferingId(specs);
  const selected =
    downloaded.find((item) => item.model === preferred) ?? downloaded[0];
  return selected?.spec ?? null;
}

export function advertisedA2vDurations(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  offering: OfferingId | null | undefined,
  resolution: A2vResolution,
): number[] {
  const spec = specForOffering(specs, offering);
  return spec == null ? [] : a2vDurationsFromSpec(spec, resolution);
}

export function a2vAspectRatios(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  offering: OfferingId | null | undefined,
  resolution: A2vResolution,
): VideoGenerationAspectRatio[] {
  const spec = specForOffering(specs, offering);
  const ratios =
    spec?.a2v_supported_resolutions_durations?.[resolution]?.aspect_ratios ?? [];
  return ratios.filter(isVideoGenerationAspectRatio);
}

export function a2vResolutionsForOffering(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  offering: OfferingId | null | undefined,
): A2vResolution[] {
  const spec = specForOffering(specs, offering);
  if (spec == null) return [];
  return A2V_RESOLUTIONS.filter(
    (resolution) => a2vDurationsFromSpec(spec, resolution).length > 0,
  );
}

/** True when any downloaded offering advertises A2V at 540p/24. */
export function hasAdvertisedA2vCell(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
): boolean {
  return (specs?.downloaded_local_models ?? []).some(
    (item) => a2vDurationsAt540p24FromSpec(item.spec).length > 0,
  );
}

export function a2vTrimCapSeconds(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  offering?: OfferingId | null,
): number | null {
  const advertised = advertisedA2vDurationsAt540p24(specs, offering);
  return advertised.length === 0 ? null : advertised[advertised.length - 1];
}

/** Longest advertised A2V duration for this offering, across resolutions. */
export function a2vLongestCellSeconds(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  offering?: OfferingId | null,
): number | null {
  let longest: number | null = null;
  for (const resolution of a2vResolutionsForOffering(specs, offering)) {
    const cap = advertisedA2vDurations(specs, offering, resolution).at(-1);
    if (cap == null) continue;
    if (longest == null || cap > longest) longest = cap;
  }
  return longest;
}

/**
 * Seconds of this clip the selected resolution will generate.
 * Inside the cell, the whole clip. Past this cell but inside a longer one,
 * only that cell's max — the opening prefix. Past every cell, under 2s, or
 * with no cell, nothing.
 */
export function a2vEffectiveAudioSeconds(
  audioDurationSeconds: number,
  advertised: readonly number[],
  longestCellSeconds: number | null,
): number | null {
  if (advertised.length === 0) return null;
  if (audioDurationSeconds < A2V_MIN_AUDIO_SECONDS) return null;
  const cap = advertised[advertised.length - 1];
  if (cap == null) return null;
  if (audioDurationSeconds < cap + A2V_CAP_TOLERANCE_SECONDS) {
    return audioDurationSeconds;
  }
  if (
    longestCellSeconds == null ||
    longestCellSeconds <= cap ||
    audioDurationSeconds >= longestCellSeconds + A2V_CAP_TOLERANCE_SECONDS
  ) {
    return null;
  }
  return cap;
}

/**
 * Integer-cell frame count for the advertised cap, mirroring backend
 * `frame_math.compute_num_frames`. Advertised cells are the hardware envelope,
 * not snap targets.
 */
export function computeA2vMaxFrames(
  maxAdvertisedSeconds: number,
  fps: number,
): number {
  return Math.max(9, Math.floor((maxAdvertisedSeconds * fps) / 8) * 8 + 1);
}

/**
 * Floor probed audio onto the 8k+1 video grid so video stays <= audio (API
 * Distilled/TIA2V `round_up=False`). Returns null when the floored frames
 * exceed the machine/resolution envelope. Mirrors backend
 * `frame_math.num_frames_for_audio_duration` without an ltxv-api dependency.
 */
export function numFramesForAudioDuration(
  durationSeconds: number,
  fps: number,
  maxFrames: number,
): number | null {
  const latentSteps = Math.max(
    1,
    Math.floor((durationSeconds * fps - 1) / 8),
  );
  const numFrames = latentSteps * 8 + 1;
  return numFrames > maxFrames ? null : numFrames;
}

/**
 * Client-side envelope check: 2s minimum, reject at/over cap +0.1s, then floor
 * to 8k+1 and refuse when the frames exceed the advertised max. Null means
 * "not generatable" (too short, too long, or no A2V cell).
 */
export function a2vNumFramesForAudio(
  audioDurationSeconds: number,
  advertised: readonly number[],
  fps: number = A2V_FPS,
): number | null {
  if (advertised.length === 0) return null;
  if (audioDurationSeconds < A2V_MIN_AUDIO_SECONDS) return null;
  const maxAdvertised = Math.max(...advertised);
  if (audioDurationSeconds >= maxAdvertised + A2V_CAP_TOLERANCE_SECONDS) {
    return null;
  }
  return numFramesForAudioDuration(
    audioDurationSeconds,
    fps,
    computeA2vMaxFrames(maxAdvertised, fps),
  );
}

/**
 * Badge label from persisted A2V frames: integers as-is, otherwise at most 2
 * decimals with trailing zeros trimmed (matches AssetPreviewModal).
 */
export function formatA2vDurationBadge(numFrames: number, fps: number): string {
  const seconds = numFrames / fps;
  const label = Number.isInteger(seconds)
    ? String(seconds)
    : seconds.toFixed(2).replace(/\.?0+$/, "");
  return `${label}s`;
}
