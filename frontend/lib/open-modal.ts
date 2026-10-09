/**
 * Radix Dialog 1.1.23 writes `role="dialog"` and never `aria-modal`.
 * Other overlays set `aria-modal="true"`.
 */
export const OPEN_MODAL_SELECTOR = '[aria-modal="true"], [role="dialog"]';

export function hasOpenModal(root: {
  querySelector(selector: string): unknown;
}): boolean {
  return root.querySelector(OPEN_MODAL_SELECTOR) != null;
}
