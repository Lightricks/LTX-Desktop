type TopEdge = { top: number };

export function computeGridScrollMargin(
  gridRect: TopEdge,
  scrollRect: TopEdge,
  scrollTop: number,
): number {
  return gridRect.top - scrollRect.top + scrollTop;
}
