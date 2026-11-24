const MINUTE_S = 60;
const HOUR_S = 60 * MINUTE_S;
const DAY_S = 24 * HOUR_S;
const MONTH_S = 30 * DAY_S;
const YEAR_S = 365 * DAY_S;

function plural(count: number, unit: string): string {
  return count === 1 ? `1 ${unit} ago` : `${count} ${unit}s ago`;
}

/** Compact relative timestamp for the results meta bar. No date-fns. */
export function formatRelativeTime(
  timestampMs: number,
  nowMs: number = Date.now(),
): string {
  const seconds = Math.max(0, Math.round((nowMs - timestampMs) / 1000));
  if (seconds < 45) return "just now";
  if (seconds < HOUR_S) return plural(Math.round(seconds / MINUTE_S), "minute");
  if (seconds < DAY_S) return plural(Math.round(seconds / HOUR_S), "hour");
  if (seconds < MONTH_S) return plural(Math.round(seconds / DAY_S), "day");
  if (seconds < YEAR_S) return plural(Math.round(seconds / MONTH_S), "month");
  return plural(Math.round(seconds / YEAR_S), "year");
}
