export type FeaturePreviewAudio = {
  muted: boolean;
  volume: number;
};

export type FeaturePreviewCapabilityProbe = {
  canPlayType(mime: string): string;
};

/** Home cards scroll inside this element, not the window. Observer root SSOT. */
export const HOME_SCROLL_ROOT_ATTR = "data-home-scroll-root";

export function homeMediaObserverRoot(element: Element): Element | null {
  return element.closest(`[${HOME_SCROLL_ROOT_ATTR}]`);
}

export function previewMimeType(url: string): string | null {
  const path = url.split(/[?#]/, 1)[0] ?? url;
  if (path.endsWith(".webm")) return "video/webm";
  if (path.endsWith(".mp4")) return "video/mp4";
  return null;
}

/**
 * Read `canPlayType` from a reused probe. A missing probe (SSR / Node tests)
 * is unsupported, matching browsers that cannot decode the clip.
 */
export function featurePreviewCanPlayType(
  mime: string,
  probe: FeaturePreviewCapabilityProbe | null,
): string {
  return probe?.canPlayType(mime) ?? "";
}

function createBrowserFeaturePreviewProbe(): FeaturePreviewCapabilityProbe | null {
  if (typeof document === "undefined") return null;
  return document.createElement("video");
}

/** One capability probe for the module lifetime. Never allocate during render. */
const browserFeaturePreviewProbe = createBrowserFeaturePreviewProbe();

export function browserFeaturePreviewCanPlayType(mime: string): string {
  return featurePreviewCanPlayType(mime, browserFeaturePreviewProbe);
}

/**
 * iPhone Safari cannot decode Home `.webm` previews. Skip attaching `src`
 * so the sibling poster image remains the visible fallback.
 */
export function canAttachFeaturePreviewSrc(
  url: string,
  canPlayType: (mime: string) => string,
): boolean {
  const mime = previewMimeType(url);
  if (mime == null) return true;
  return canPlayType(mime) !== "";
}

/**
 * Whether the feature preview should carry a looping `src`. A failed preview,
 * reduced-motion preference, or viewport miss falls back to the poster.
 */
export function shouldPlayFeaturePreview({
  previewFailed,
  reducedMotion,
  hasBeenVisible,
  inView,
}: {
  previewFailed: boolean;
  reducedMotion: boolean;
  hasBeenVisible: boolean;
  inView: boolean;
}): boolean {
  return !previewFailed && !reducedMotion && hasBeenVisible && inView;
}

/** Home feature previews never play sound. */
export function silenceFeaturePreview(video: FeaturePreviewAudio): void {
  video.muted = true;
  video.volume = 0;
}
