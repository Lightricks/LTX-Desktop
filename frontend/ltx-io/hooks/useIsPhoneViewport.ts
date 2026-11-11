import { useSyncExternalStore } from "react";

/** Keep in sync with the `@media (width < 768px)` phone rules of the shared modals. */
export const PHONE_VIEWPORT_QUERY = "(max-width: 767px)";

function subscribe(onChange: () => void): () => void {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return () => {};
  }
  const query = window.matchMedia(PHONE_VIEWPORT_QUERY);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

function getSnapshot(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia(PHONE_VIEWPORT_QUERY).matches
    : false;
}

/** True while the window is as narrow as a phone. It follows a resize or a turn of the device. */
export function useIsPhoneViewport(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, () => false);
}
