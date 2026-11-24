/** Tile width that yields a readable frame at the 48px compact track height. */
const FRAME_SLOT_PX = 36;
const MIN_FRAME_COUNT = 6;
const MAX_FRAME_COUNT = 24;

export function videoFilmstripFrameCount(trackWidth: number): number {
  if (trackWidth <= 0) return MIN_FRAME_COUNT;
  return Math.max(
    MIN_FRAME_COUNT,
    Math.min(MAX_FRAME_COUNT, Math.round(trackWidth / FRAME_SLOT_PX)),
  );
}

/** Sample the middle of each tile so the first/last frames aren't the freeze-frame edges. */
export function videoFilmstripTimes(
  durationSeconds: number,
  count: number,
): number[] {
  if (durationSeconds <= 0 || count <= 0) return [];
  const last = Math.max(0, durationSeconds - 0.04);
  return Array.from({ length: count }, (_, index) => {
    const time = ((index + 0.5) / count) * durationSeconds;
    return Math.min(last, Math.max(0, time));
  });
}
