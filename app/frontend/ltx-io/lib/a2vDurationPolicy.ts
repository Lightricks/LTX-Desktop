import {
  type DisabledVideoGenerationOption,
  type DroppedFastAspectCell,
  type DroppedFastCell,
  type OfferingId,
  type VideoGenerationAspectRatio,
  type VideoGenerationModelSpecItem,
  type VideoGenerationModelSpecsResponse,
  defaultOfferingId,
  disabledOptionsFromDroppedCells,
  isVideoGenerationAspectRatio,
  specForAspect,
} from "../../lib/video-generation-model-specs.ts";

export const A2V_RESOLUTION = "540p" as const;
export const A2V_FPS = 24 as const;
export const A2V_RESOLUTIONS = ["270p", "360p", "540p", "720p", "1080p", "1440p", "2160p"] as const;
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
  return (
    typeof value === "string" && (A2V_RESOLUTIONS as readonly string[]).includes(value)
  );
}

function a2vDurationsFromSpec(spec: A2vSpec, resolution: A2vResolution): number[] {
  const kept =
    spec.a2v_supported_resolutions_durations?.[resolution]?.fps_to_durations?.[
      String(A2V_FPS)
    ] ?? [];
  return [...kept].sort((a, b) => a - b);
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
    return selected ? a2vDurationsFromSpec(selected.spec, A2V_RESOLUTION) : [];
  }
  const preferred = defaultOfferingId(specs);
  const selected = downloaded.find((item) => item.model === preferred) ?? downloaded[0];
  return selected ? a2vDurationsFromSpec(selected.spec, A2V_RESOLUTION) : [];
}

function specForOffering(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  offering?: OfferingId | null,
  aspectRatio?: string | null,
): A2vSpec | null {
  const downloaded = specs?.downloaded_local_models ?? [];
  const baseSpec =
    offering != null
      ? (downloaded.find((item) => item.model === offering)?.spec ?? null)
      : ((downloaded.find((item) => item.model === defaultOfferingId(specs)) ?? downloaded[0])
          ?.spec ?? null);
  return baseSpec == null ? null : specForAspect(baseSpec, aspectRatio);
}

/** Kept durations plus dropped ones: the model's ceiling, not the budget filter. */
export function staticA2vDurations(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  offering: OfferingId | null | undefined,
  resolution: A2vResolution,
  aspectRatio?: string | null,
): number[] {
  const spec = specForOffering(specs, offering, aspectRatio);
  if (spec == null) return [];
  const kept =
    spec.a2v_supported_resolutions_durations?.[resolution]?.fps_to_durations?.[
      String(A2V_FPS)
    ] ?? [];
  const dropped = (spec.dropped_cells ?? [])
    .filter(
      (cell) => cell.audio && cell.resolution === resolution && cell.fps === A2V_FPS,
    )
    .map((cell) => cell.duration);
  return [...new Set([...kept, ...dropped])].sort((left, right) => left - right);
}

export function staticA2vLongestCellSeconds(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  offering?: OfferingId | null,
  aspectRatio?: string | null,
): number | null {
  let longest: number | null = null;
  for (const resolution of A2V_RESOLUTIONS) {
    const cap = staticA2vDurations(specs, offering, resolution, aspectRatio).at(-1);
    if (cap == null) continue;
    if (longest == null || cap > longest) longest = cap;
  }
  return longest;
}

export function advertisedA2vDurations(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  offering: OfferingId | null | undefined,
  resolution: A2vResolution,
  aspectRatio?: string | null,
): number[] {
  const spec = specForOffering(specs, offering, aspectRatio);
  return spec == null ? [] : a2vDurationsFromSpec(spec, resolution);
}

export function a2vAspectRatios(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  offering: OfferingId | null | undefined,
  resolution: A2vResolution,
): VideoGenerationAspectRatio[] {
  const spec = specForOffering(specs, offering);
  if (spec == null) return [];
  const direct = spec.a2v_supported_resolutions_durations?.[resolution]?.aspect_ratios;
  if (direct != null) {
    return direct.filter(isVideoGenerationAspectRatio);
  }
  for (const aspectSpec of Object.values(spec.aspect_specs ?? {})) {
    const ratios = aspectSpec.a2v_supported_resolutions_durations?.[resolution]?.aspect_ratios;
    if (ratios != null) {
      return ratios.filter(isVideoGenerationAspectRatio);
    }
  }
  return [];
}

export function a2vResolutionsForOffering(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  offering: OfferingId | null | undefined,
  aspectRatio?: string | null,
): A2vResolution[] {
  const spec = specForOffering(specs, offering, aspectRatio);
  if (spec == null) return [];
  return A2V_RESOLUTIONS.filter(
    (resolution) => a2vDurationsFromSpec(spec, resolution).length > 0,
  );
}

/**
 * Withheld A2V resolutions for disabled-with-tooltip pickers: dropped
 * everywhere at 24fps, kept nowhere, messaged from the cheapest dropped
 * cell. Empty without specs.
 */
export function a2vDisabledResolutionsForOffering(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  offering: OfferingId | null | undefined,
  aspectRatio?: string | null,
): DisabledVideoGenerationOption<A2vResolution>[] {
  const spec = specForOffering(specs, offering, aspectRatio);
  if (spec == null) return [];
  const kept = a2vResolutionsForOffering(specs, offering, aspectRatio);
  const cells = (spec.dropped_cells ?? []).filter(
    (cell): cell is DroppedFastCell & { resolution: A2vResolution } =>
      cell.audio && isA2vResolution(cell.resolution),
  );
  return disabledOptionsFromDroppedCells(
    cells,
    (cell) => cell.resolution,
    kept,
    (a, b) => A2V_RESOLUTIONS.indexOf(a) - A2V_RESOLUTIONS.indexOf(b),
  );
}

/**
 * Shapes the current audio cannot run at this resolution, for the
 * disabled-with-tooltip aspect picker. Mirrors the t2v/i2v aspect mapping
 * (audio + current res/fps/duration), except the duration match is
 * monotonic, not exact: the job is `effectiveSeconds` long (a float), while
 * cells are integer seconds, so a shape refused at any cell at or under the
 * job length refuses the job (the budget only grows with duration). The
 * refusal names the longest such cell — nearest the job. Null while the
 * length is unknown: unknown stays enabled and create-time enforcement
 * names the refusal instead.
 */
export function a2vDisabledAspectsForOffering(
  specs: VideoGenerationModelSpecsResponse | null | undefined,
  offering: OfferingId | null | undefined,
  resolution: A2vResolution,
  effectiveSeconds: number | null,
): DisabledVideoGenerationOption<VideoGenerationAspectRatio>[] {
  const spec = specForOffering(specs, offering);
  if (spec == null || effectiveSeconds == null) return [];
  const aspectSpecs = spec.aspect_specs;
  if (aspectSpecs) {
    const out: DisabledVideoGenerationOption<VideoGenerationAspectRatio>[] = [];
    for (const [ratio, aspectSpec] of Object.entries(aspectSpecs)) {
      if (!isVideoGenerationAspectRatio(ratio) || !aspectSpec) continue;
      let chosen: DroppedFastCell | null = null;
      for (const cell of aspectSpec.dropped_cells ?? []) {
        if (
          !cell.audio ||
          cell.resolution !== resolution ||
          cell.fps !== A2V_FPS ||
          cell.duration > effectiveSeconds
        ) {
          continue;
        }
        if (chosen == null || cell.duration > chosen.duration) {
          chosen = cell;
        }
      }
      if (chosen != null) {
        out.push({
          value: ratio,
          message: chosen.message,
          ...(chosen.readout ? { readout: chosen.readout } : {}),
        });
      }
    }
    return out;
  }
  const byAspect = new Map<VideoGenerationAspectRatio, DroppedFastAspectCell>();
  for (const cell of spec.dropped_aspect_cells ?? []) {
    if (
      !cell.audio ||
      cell.resolution !== resolution ||
      cell.fps !== A2V_FPS ||
      cell.duration > effectiveSeconds
    ) {
      continue;
    }
    const aspectRatio = cell.aspectRatio;
    if (!isVideoGenerationAspectRatio(aspectRatio)) continue;
    const current = byAspect.get(aspectRatio);
    if (current == null || cell.duration > current.duration) {
      byAspect.set(aspectRatio, cell);
    }
  }
  return [...byAspect.entries()].map(([value, cell]) => ({
    value,
    message: cell.message,
    ...(cell.readout ? { readout: cell.readout } : {}),
  }));
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

/**
 * Smallest kept cell whose 8k+1 count covers the heard frames.
 * 241 frames is the 10s cell even when a 20s cell is also offered, so a 10.2s
 * clip is not priced as 20s.
 */
export function a2vDurationForFrames(
  durations: readonly number[],
  frames: number | null,
): number | null {
  if (frames == null) return null;
  return (
    durations.find((duration) => frames <= computeA2vMaxFrames(duration, A2V_FPS)) ?? null
  );
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

export type A2vAudioFit = {
  /** Seconds the backend would hear, or null when the static ceiling rejects the clip. */
  heardSeconds: number | null;
  /** Floored 8k+1 frames for `heardSeconds`, or null when nothing runs. */
  frames: number | null;
  /** Those frames do not fit a budget-kept cell, so Generate must stay off. */
  pastKept: boolean;
  keptMaxSeconds: number | null;
};

/**
 * Heard seconds follow the static ceiling. Generate is allowed only when the
 * floored 8k+1 count still fits a budget-kept cell. 10.2s at 24fps is 241
 * frames, the same as a kept 10s cell, so a seconds tolerance would refuse a
 * clip the backend accepts. A prefix of a dropped ceiling is `pastKept`.
 * The form warns on the smallest kept cell that covers those frames, so it
 * can warn when the backend stores no risk for the exact frame count.
 */
export function resolveA2vAudio(input: {
  audioSeconds: number;
  keptSeconds: readonly number[];
  ceilingSeconds: readonly number[];
  longestCeilingSeconds: number | null;
}): A2vAudioFit {
  const fps = A2V_FPS;
  const keptMaxSeconds =
    input.keptSeconds.length === 0
      ? null
      : (input.keptSeconds[input.keptSeconds.length - 1] ?? null);
  const ceilingMax =
    input.ceilingSeconds.length === 0
      ? null
      : (input.ceilingSeconds[input.ceilingSeconds.length - 1] ?? null);
  if (input.audioSeconds < A2V_MIN_AUDIO_SECONDS || ceilingMax == null) {
    return { heardSeconds: null, frames: null, pastKept: false, keptMaxSeconds };
  }
  const heardSeconds = a2vEffectiveAudioSeconds(
    input.audioSeconds,
    input.ceilingSeconds,
    input.longestCeilingSeconds,
  );
  if (heardSeconds == null || heardSeconds >= ceilingMax + A2V_CAP_TOLERANCE_SECONDS) {
    return { heardSeconds: null, frames: null, pastKept: false, keptMaxSeconds };
  }
  const frames = numFramesForAudioDuration(
    heardSeconds,
    fps,
    computeA2vMaxFrames(ceilingMax, fps),
  );
  if (frames == null) {
    return { heardSeconds: null, frames: null, pastKept: false, keptMaxSeconds };
  }
  // Frames past the kept cell's own count are a client hard stop. The server
  // prices the exact count and can still accept a few extra frames when that
  // cell has memory headroom. The form does not guess that headroom.
  const pastKept =
    keptMaxSeconds == null || frames > computeA2vMaxFrames(keptMaxSeconds, fps);
  return { heardSeconds, frames, pastKept, keptMaxSeconds };
}

/**
 * Integer-cell frame count for the advertised cap, mirroring backend
 * `frame_math.compute_num_frames`. Advertised cells are the hardware envelope,
 * not snap targets.
 */
export function computeA2vMaxFrames(maxAdvertisedSeconds: number, fps: number): number {
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
  const latentSteps = Math.max(1, Math.floor((durationSeconds * fps - 1) / 8));
  const numFrames = latentSteps * 8 + 1;
  return numFrames > maxFrames ? null : numFrames;
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
