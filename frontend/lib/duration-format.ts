/** Duration labels for the ETA: one splitter, three wordings. */

export const FINISHING_LABEL = "Finishing…";

type DurationParts =
  | { unit: "sec"; seconds: number }
  | { unit: "min"; minutes: number }
  | { unit: "hr"; hours: number; minutes: number };

type Wording = {
  prefix: string;
  sec: string;
  min: string;
  hr: string;
  suffix: string;
};

const SHORT: Wording = { prefix: "~", sec: "s", min: "m", hr: "h", suffix: " left" };
const LONG: Wording = { prefix: "About ", sec: " sec", min: " min", hr: " hr", suffix: " left" };
const ESTIMATE: Wording = { ...LONG, suffix: "" };

function isUsable(seconds: number | null | undefined): seconds is number {
  return seconds != null && Number.isFinite(seconds) && seconds >= 0;
}

function splitDuration(seconds: number, roundSeconds: (value: number) => number): DurationParts {
  if (seconds < 60) return { unit: "sec", seconds: roundSeconds(seconds) };
  const totalMinutes = Math.max(1, Math.round(seconds / 60));
  if (totalMinutes < 60) return { unit: "min", minutes: totalMinutes };
  return { unit: "hr", hours: Math.floor(totalMinutes / 60), minutes: totalMinutes % 60 };
}

function render(parts: DurationParts, wording: Wording): string {
  const { prefix, suffix } = wording;
  switch (parts.unit) {
    case "sec":
      return `${prefix}${parts.seconds}${wording.sec}${suffix}`;
    case "min":
      return `${prefix}${parts.minutes}${wording.min}${suffix}`;
    case "hr":
      return parts.minutes === 0
        ? `${prefix}${parts.hours}${wording.hr}${suffix}`
        : `${prefix}${parts.hours}${wording.hr} ${parts.minutes}${wording.min}${suffix}`;
  }
}

/** Remaining seconds count down, so a partial five seconds rounds up. */
const roundUpToFive = (seconds: number) => Math.max(5, Math.ceil(seconds / 5) * 5);

function remainingLabel(seconds: number | null | undefined, wording: Wording): string | null {
  if (!isUsable(seconds)) return null;
  if (seconds === 0) return FINISHING_LABEL;
  return render(splitDuration(seconds, roundUpToFive), wording);
}

/** "~2m left", for a tight label such as the queue row title. */
export function formatShortRemainingTime(seconds: number | null | undefined): string | null {
  return remainingLabel(seconds, SHORT);
}

/** "About 2 min left". */
export function formatRemainingTime(seconds: number | null | undefined): string | null {
  return remainingLabel(seconds, LONG);
}

/** "About 2 min", the whole-run estimate shown before a run starts. */
export function formatEstimatedDuration(seconds: number | null | undefined): string | null {
  if (!isUsable(seconds) || seconds === 0) return null;
  const roundToFive = (value: number) => Math.max(10, Math.round(value / 5) * 5);
  return render(splitDuration(seconds, roundToFive), ESTIMATE);
}
