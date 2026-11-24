const SWIPE_THRESHOLD_PX = 56;
const SWIPE_IGNORED_TARGET_SELECTOR =
  "[data-lightbox-chrome], [data-lightbox-controls]";

export function isLightboxGallerySwipePointer(pointerType: string): boolean {
  return pointerType === "touch" || pointerType === "pen";
}

export function isLightboxSwipeIgnoredTarget(target: EventTarget | null): boolean {
  return target instanceof Element
    ? target.closest(SWIPE_IGNORED_TARGET_SELECTOR) !== null
    : false;
}

export function swipeGalleryDirection(
  startX: number,
  startY: number,
  endX: number,
  endY: number,
  thresholdPx = SWIPE_THRESHOLD_PX,
): "previous" | "next" | null {
  const deltaX = endX - startX;
  const deltaY = endY - startY;
  if (Math.abs(deltaX) < thresholdPx) return null;
  if (Math.abs(deltaX) <= Math.abs(deltaY)) return null;
  return deltaX > 0 ? "previous" : "next";
}
