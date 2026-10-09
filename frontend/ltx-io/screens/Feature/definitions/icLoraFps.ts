import cap from "../../../../../shared/ic-lora-input-cap.json" with { type: "json" };

/** Supported output rates, highest-last in the shared file. */
export const IC_LORA_SUPPORTED_FPS = cap.supportedFps;

export const IC_LORA_FPS_MATCH_TOLERANCE = cap.fpsMatchTolerance;

export function icLoraMaxInputSeconds(fps: number): number {
  return Math.min(cap.maxInputVideoSeconds, cap.maxInputFrames / fps);
}

export function formatIcLoraSeconds(seconds: number): string {
  const rounded = Math.round(seconds * 10) / 10;
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
}

function formatFps(rate: number): string {
  if (Number.isInteger(rate)) return String(rate);
  const rounded = Math.round(rate * 1000) / 1000;
  return String(rounded);
}

/** Supported rate closest to the source, never more than half a frame above it. */
export function icLoraOriginalFps(sourceFps: number): number {
  const eligible = IC_LORA_SUPPORTED_FPS.filter(
    (rate) => rate <= sourceFps + IC_LORA_FPS_MATCH_TOLERANCE,
  );
  if (eligible.length === 0) return sourceFps;
  return eligible.reduce((best, rate) => {
    const distance = Math.abs(rate - sourceFps);
    const bestDistance = Math.abs(best - sourceFps);
    if (distance < bestDistance) return rate;
    if (distance === bestDistance) return Math.min(best, rate);
    return best;
  });
}

export type IcLoraFpsOption = {
  value: number;
  label: string;
  original: boolean;
};

/** Every supported rate below Original, then Original, lowest first. */
export function icLoraFpsOptions(sourceFps: number): IcLoraFpsOption[] {
  const original = icLoraOriginalFps(sourceFps);
  const below = IC_LORA_SUPPORTED_FPS.filter((rate) => rate < original).sort(
    (left, right) => left - right,
  );
  return [...below, original].map((value) => ({
    value,
    original: value === original,
    label: value === original ? `${formatFps(value)} (Original)` : formatFps(value),
  }));
}

/** The rate the output uses: the chosen one, or Original for the clip, or the lowest supported. */
export function icLoraOutputFps(fps: number | null, videoFps: number | null): number {
  if (fps != null) return fps;
  if (videoFps != null) return icLoraOriginalFps(videoFps);
  return IC_LORA_SUPPORTED_FPS[0] ?? 24;
}

/** Original, and any rate the clip no longer offers, stay unset. */
export function icLoraSelectedFps(fps: number | null, videoFps: number | null): number | null {
  if (fps == null || videoFps == null) return fps;
  const offered = icLoraFpsOptions(videoFps);
  if (fps === icLoraOriginalFps(videoFps) || !offered.some((option) => option.value === fps)) {
    return null;
  }
  return fps;
}
