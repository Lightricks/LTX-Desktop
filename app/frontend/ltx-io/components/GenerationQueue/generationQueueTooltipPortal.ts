/** Escape the queue panel stacking context so tooltips paint above the popover. */
export const generationQueueTooltipPortal =
  typeof document !== "undefined" ? document.body : null;
