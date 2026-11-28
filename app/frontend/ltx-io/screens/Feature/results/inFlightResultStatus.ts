import { formatRemainingTime } from "../../../../lib/duration-format.ts";

export type InFlightResultStatus = {
  title?: string;
  body?: string;
  showDots: boolean;
};

/**
 * In-flight result copy, matching studio InlinePreview: dots until a real
 * percent exists, then the percent replaces the dots.
 */
export function inFlightResultStatus(args: {
  status: string;
  progressPercent: number | undefined;
  remainingSeconds?: number | null;
}): InFlightResultStatus {
  if (args.status === "cancelling") {
    return { title: "Cancellation in progress", showDots: false };
  }
  const body = formatRemainingTime(args.remainingSeconds) ?? undefined;
  if (args.progressPercent !== undefined) {
    return {
      title: `${Math.round(args.progressPercent)}%`,
      ...(body ? { body } : {}),
      showDots: false,
    };
  }
  return {
    ...(body ? { body } : {}),
    showDots: true,
  };
}
