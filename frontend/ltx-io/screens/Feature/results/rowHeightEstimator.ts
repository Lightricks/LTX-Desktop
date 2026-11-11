const FRAME_ASPECT = 9 / 16;
// The height of a row besides its video frame (metadata row and the gap under it). This
// is a guess for the first rows only. Measured rows replace it.
const INITIAL_ROW_OVERHEAD_PX = 56;
const FALLBACK_ROW_PX = 400;

type RowKey = string | number | bigint;

/**
 * Estimates the height of a row that was never on screen. The frame height follows the
 * feed width. The rest (metadata row) is the average of the rows measured so far.
 * A close estimate keeps the scroll length steady while the user scrolls and rows are
 * measured. A far estimate makes the scrollbar grow and the content jump.
 */
export function createRowHeightEstimator() {
  const overheadByKey = new Map<RowKey, number>();
  let overheadSum = 0;
  return {
    /** Records a measured row. A row that is measured again replaces its old value. */
    record(key: RowKey, measuredHeight: number, feedWidth: number) {
      if (measuredHeight <= 0 || feedWidth <= 0) return;
      const overhead = measuredHeight - feedWidth * FRAME_ASPECT;
      overheadSum += overhead - (overheadByKey.get(key) ?? 0);
      overheadByKey.set(key, overhead);
    },
    estimate(feedWidth: number): number {
      if (feedWidth <= 0) return FALLBACK_ROW_PX;
      const overhead =
        overheadByKey.size > 0
          ? overheadSum / overheadByKey.size
          : INITIAL_ROW_OVERHEAD_PX;
      return feedWidth * FRAME_ASPECT + overhead;
    },
  };
}
