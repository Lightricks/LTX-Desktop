type Rect = { left: number; top: number; right: number; bottom: number };
type Size = { width: number; height: number };

const EDGE_MARGIN = 8;
const GAP = 8;

/**
 * Where a popover goes, in viewport pixels. It prefers the left of the anchor, so
 * it covers the form and not the preview. With no room on the left (fullscreen)
 * it opens on the right and stays on screen.
 */
export function placeBesideAnchor(
  anchor: Rect,
  popover: Size,
  viewport: Size,
): { left: number; top: number } {
  const leftOfAnchor = anchor.left - GAP - popover.width;
  const left =
    leftOfAnchor >= EDGE_MARGIN
      ? leftOfAnchor
      : Math.min(anchor.right + GAP, viewport.width - popover.width - EDGE_MARGIN);
  const top = Math.min(anchor.top, viewport.height - popover.height - EDGE_MARGIN);
  return { left: Math.max(EDGE_MARGIN, left), top: Math.max(EDGE_MARGIN, top) };
}
