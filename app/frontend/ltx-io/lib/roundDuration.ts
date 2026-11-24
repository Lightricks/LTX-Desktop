/** Wire-format duration rounding. Kept out of presentation so UI tweaks cannot change the API. */
export function roundDurationSeconds(seconds: number): number {
  return Math.round(seconds * 10) / 10;
}
