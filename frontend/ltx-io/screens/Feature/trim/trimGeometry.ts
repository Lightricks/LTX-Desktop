import { MIN_TRIM_DURATION_SECONDS } from "./trimConstants.ts";

export type TrimRange = {
  startSec: number;
  endSec: number;
};

export function timeToPixels(time: number, pixelsPerSecond: number): number {
  return time * pixelsPerSecond;
}

/** Filmstrip/waveform fills the bar; In/Out handles overlay the frames. */
export function trimMediaWidth(containerWidth: number): number {
  return Math.max(1, containerWidth);
}

/** Selection box including handles, clamped so rounded caps stay on the bar. */
export function trimSelectionRect(
  startPx: number,
  endPx: number,
  handleWidth: number,
  containerWidth: number,
): { left: number; width: number } {
  const left = Math.max(0, startPx - handleWidth);
  const right = Math.min(containerWidth, endPx + handleWidth);
  return { left, width: Math.max(0, right - left) };
}

export function pixelsToTime(pixels: number, pixelsPerSecond: number): number {
  return pixels / pixelsPerSecond;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/**
 * Open with the first third selected (capped at the model's max) so users can
 * expand or shrink from a sensible starting range instead of the full clip.
 * Ported from LTX.io `AudioTrimRange`, with the selection floor applied so the
 * modal never opens on a range its own Done button would reject.
 */
export function initialTrimRange(
  durationSeconds: number,
  maxDurationSeconds: number,
  minDurationSeconds: number = MIN_TRIM_DURATION_SECONDS,
): TrimRange {
  const widest = Math.min(maxDurationSeconds, durationSeconds);
  return {
    startSec: 0,
    endSec: clamp(
      durationSeconds / 3,
      Math.min(minDurationSeconds, widest),
      widest,
    ),
  };
}

/**
 * Done sends `[startSec, endSec]` to the backend trimmer, so it must hold the
 * same `[min, cap]` window the drag handles enforce.
 */
export function isApplicableTrimRange(
  range: TrimRange,
  bounds: {
    durationSeconds: number;
    maxDurationSeconds: number;
    minDurationSeconds?: number;
  },
): boolean {
  const {
    durationSeconds,
    maxDurationSeconds,
    minDurationSeconds = MIN_TRIM_DURATION_SECONDS,
  } = bounds;
  const selected = range.endSec - range.startSec;
  return (
    range.startSec >= 0 &&
    range.endSec <= durationSeconds &&
    selected >= Math.min(minDurationSeconds, durationSeconds) &&
    selected <= maxDurationSeconds
  );
}

/**
 * Keeps a range inside `[0, durationSeconds]` and within the min/max window.
 * `anchor` says which edge the user is holding: the opposite edge stays put.
 */
export function clampTrimRange(
  range: TrimRange,
  bounds: {
    durationSeconds: number;
    maxDurationSeconds: number;
    minDurationSeconds?: number;
    anchor: "start" | "end" | "both";
  },
): TrimRange {
  const {
    durationSeconds,
    maxDurationSeconds,
    minDurationSeconds = MIN_TRIM_DURATION_SECONDS,
    anchor,
  } = bounds;

  // A misconfigured min above max would otherwise make every range invalid.
  const minWindow = Math.min(minDurationSeconds, maxDurationSeconds);
  const maxWindow = Math.min(maxDurationSeconds, durationSeconds);

  if (anchor === "both") {
    const width = clamp(range.endSec - range.startSec, minWindow, maxWindow);
    const startSec = clamp(range.startSec, 0, durationSeconds - width);
    return { startSec, endSec: startSec + width };
  }

  if (anchor === "start") {
    const endSec = clamp(range.endSec, minWindow, durationSeconds);
    const startSec = clamp(
      range.startSec,
      Math.max(0, endSec - maxWindow),
      endSec - minWindow,
    );
    return { startSec, endSec };
  }

  const startSec = clamp(range.startSec, 0, durationSeconds - minWindow);
  const endSec = clamp(
    range.endSec,
    startSec + minWindow,
    Math.min(durationSeconds, startSec + maxWindow),
  );
  return { startSec, endSec };
}

/**
 * Moves the selection so it starts at the playhead.
 * Returns null when the clip after the needle is shorter than the minimum
 * window: there is no legal place for the selection to start, so it stays put.
 * When that tail is long enough but shorter than the current selection, the
 * selection shrinks to the tail instead of running past the clip.
 */
export function snapSelectionToNeedle(
  needleSec: number,
  range: TrimRange,
  bounds: {
    durationSeconds: number;
    maxDurationSeconds: number;
    minDurationSeconds?: number;
  },
): TrimRange | null {
  const {
    durationSeconds,
    maxDurationSeconds,
    minDurationSeconds = MIN_TRIM_DURATION_SECONDS,
  } = bounds;

  if (!Number.isFinite(needleSec) || !(durationSeconds > 0)) return null;

  const minWindow = Math.min(
    minDurationSeconds,
    maxDurationSeconds,
    durationSeconds,
  );
  const maxWindow = Math.min(maxDurationSeconds, durationSeconds);
  const startSec = clamp(needleSec, 0, durationSeconds);
  const remaining = durationSeconds - startSec;
  if (remaining < minWindow) return null;

  const currentWidth = Math.max(0, range.endSec - range.startSec);
  const width = Math.min(
    Math.max(currentWidth, minWindow),
    remaining,
    maxWindow,
  );
  return {
    startSec,
    endSec: width === remaining ? durationSeconds : startSec + width,
  };
}

/** `M:SS` for the selection badge and the playback readout. */
export function formatTrimTimecode(seconds: number): string {
  const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  const totalSeconds = Math.floor(safe);
  const minutes = Math.floor(totalSeconds / 60);
  const remainder = totalSeconds % 60;
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

/** Where the playhead may travel. Omit for In/Out-window playback; pass a range for full-clip. */
export function playheadWindow(
  playheadRange: TrimRange | undefined,
  selection: TrimRange,
): TrimRange {
  return playheadRange ?? selection;
}
