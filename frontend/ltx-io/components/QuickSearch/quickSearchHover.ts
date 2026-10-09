export const QUICK_SEARCH_HOVER_INTENT_MS = 180;

export type QuickSearchPointer = {
  x: number;
  y: number;
};

/**
 * Moving right into the preview crosses other rows. Defer that switch so the
 * preview stays on the row the pointer left. Movement down the list switches
 * immediately.
 */
export function quickSearchHoverIntent(
  previous: QuickSearchPointer,
  next: QuickSearchPointer,
): "ignore" | "immediate" | "defer" {
  if (next.x === previous.x && next.y === previous.y) return "ignore";
  const dx = next.x - previous.x;
  const dy = next.y - previous.y;
  const towardPreview = Number.isFinite(dx) && dx > 4 && dx >= Math.abs(dy);
  return towardPreview ? "defer" : "immediate";
}
