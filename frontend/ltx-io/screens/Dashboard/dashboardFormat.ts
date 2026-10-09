// Wording follows the codes stored on generations.error_code.
const FAILURE_LABELS: Record<string, string> = {
  INTERRUPTED: "Interrupted",
  INPUT_MISSING: "Missing input",
  OUTPUT_MISSING: "Missing video",
  OUTPUT_UNREADABLE: "Unreadable video",
  EXECUTOR_FAILED: "Render failed",
  CAPABILITY_FAILED: "Could not start",
  LTX_INVALID_API_KEY: "API key rejected",
  LTX_API_PROMPT_EMBEDDING_FAILED: "Prompt encoding failed",
  UNKNOWN: "Unknown error",
};

const FIX_HINTS: Record<string, string> = {
  INTERRUPTED: "The app quit while this run was still going. Generate it again.",
  INPUT_MISSING: "An input file was missing when the run started. Add it again and generate.",
  OUTPUT_MISSING: "The run finished, but the video file was not on disk. Generate again.",
  OUTPUT_UNREADABLE: "The video file was written, but it could not be read. Generate again.",
  EXECUTOR_FAILED: "The run failed while rendering. Generate again.",
  CAPABILITY_FAILED:
    "This run could not start with the current model or inputs. Check them, then generate again.",
  LTX_INVALID_API_KEY:
    "The LTX API key was rejected while encoding the prompt. Update it in Settings.",
  LTX_API_PROMPT_EMBEDDING_FAILED:
    "Prompt encoding through the LTX API failed. Check the key and connection, then generate again.",
  UNKNOWN: "The run failed without an error code. Generate again.",
};

export const DASHBOARD_INFO = {
  content:
    "Finished video clips in this range, including clips you later delete from Assets. The line below is the new footage they hold: each clip's length, and for Retake and Extend only the part you edited.",
  gpu: "How long generation ran in this range. Succeeded and failed runs are included. Cancelled runs are not.",
  success:
    "Succeeded runs divided by succeeded and failed runs. Cancelled runs are listed beside the rate and are not part of it.",
  render:
    "Typical time to finish a video in the selected range, grouped by resolution, aspect ratio, and length. Includes text, image, audio, and LoRA renders. Retake and Extend are not included. Select a square for the median and the 90th percentile.",
  loras:
    "Styles used in this range, ranked by number of runs. Keep is the share of that style's clips still in your library, using the same rule as Keep rate.",
  keep: "The share of finished clips in this range that are still in your library. Deleting a clip from Assets, or deleting the generation that made it, marks it as dropped and lowers this rate. Deleted clips stay in the total.",
  usual:
    "The resolution, aspect ratio, length, and frame rate you finish most often in this range.",
} as const;

export const ACTIVITY_SERIES = [
  { id: "text-to-video", label: "Text to video" },
  { id: "image-to-video", label: "Image to video" },
  { id: "audio-to-video", label: "Audio to video" },
  { id: "lora", label: "LoRA" },
  { id: "retake", label: "Retake" },
  { id: "extend", label: "Extend" },
  { id: "ic-lora", label: "IC-LoRA" },
] as const;

export function failureLabel(code: string): string {
  return FAILURE_LABELS[code] ?? code;
}

export function fixHint(code: string): string {
  return FIX_HINTS[code] ?? "Generate again.";
}

export function formatPercent(rate: number | null | undefined): string {
  if (rate == null) return "–";
  return `${Math.round(rate * 100)}%`;
}

export function formatFootage(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
}

export function formatGpu(ms: number): string {
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 0) return minutes ? `${hours}h ${minutes}m` : `${hours}h`;
  return `${minutes}m`;
}

/** Spoken duration for hover text and the selected-cell line. */
export function formatDuration(ms: number): string {
  const total = Math.max(0, Math.round(ms / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  const parts: string[] = [];
  if (hours > 0) parts.push(`${hours}h`);
  if (minutes > 0) parts.push(`${minutes}m`);
  if (seconds > 0 || parts.length === 0) parts.push(`${seconds}s`);
  return parts.join(" ");
}

/** Compact clock time, so a full duration row fits without sideways scrolling. */
export function formatRun(ms: number): string {
  const total = Math.max(0, Math.round(ms / 1000));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

export function renderBand(ms: number): "fast" | "mid" | "slow" {
  if (ms < 60_000) return "fast";
  if (ms <= 180_000) return "mid";
  return "slow";
}

export function formatSeconds(value: number): string {
  return `${trimNumber(value)}s`;
}

const ACTIVITY_DAY = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  timeZone: "UTC",
});
const ACTIVITY_DAY_WITH_YEAR = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "numeric",
  year: "numeric",
  timeZone: "UTC",
});

/** Calendar day from the API (`YYYY-MM-DD`), kept on that day regardless of timezone. */
export function formatActivityDay(isoDay: string, withYear = false): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(isoDay);
  if (!match) return isoDay;
  const date = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3])));
  return (withYear ? ACTIVITY_DAY_WITH_YEAR : ACTIVITY_DAY).format(date);
}

function trimNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : String(Math.round(value * 10) / 10);
}
