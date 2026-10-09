/** Desktop live setting wins; Remote uses the status fetched when the feature form mounts. */
export function resolveShowManualEnhance(
  liveAutoEnhance: boolean | undefined,
  statusShowManualEnhance: boolean | undefined,
): boolean {
  if (liveAutoEnhance !== undefined) return !liveAutoEnhance;
  return statusShowManualEnhance ?? false;
}
