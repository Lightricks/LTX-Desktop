export type InFlightResultStatus = {
  title?: string;
  showDots: boolean;
};

/**
 * In-flight result copy, matching studio InlinePreview: dots until a real
 * percent exists, then the percent replaces the dots.
 */
export function inFlightResultStatus(args: {
  status: string;
  progressPercent: number | undefined;
}): InFlightResultStatus {
  if (args.status === "cancelling") {
    return { title: "Cancellation in progress", showDots: false };
  }
  if (args.progressPercent !== undefined) {
    return { title: `${Math.round(args.progressPercent)}%`, showDots: false };
  }
  return { showDots: true };
}
