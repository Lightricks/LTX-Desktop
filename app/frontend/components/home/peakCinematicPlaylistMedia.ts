/** Whether this stacked clip should attach a video `src` (active or adjacent prefetch). */
export function shouldAttachPeakVideoSrc({
  hasBeenVisible,
  inView,
  isActive,
  hasPlaylist,
  index,
  activeIndex,
  length,
}: {
  hasBeenVisible: boolean;
  inView: boolean;
  isActive: boolean;
  hasPlaylist: boolean;
  index: number;
  activeIndex: number;
  length: number;
}): boolean {
  if (!hasBeenVisible || !inView) return false;
  if (isActive) return true;
  if (!hasPlaylist || length === 0) return false;
  const next = (activeIndex + 1) % length;
  const prev = (activeIndex - 1 + length) % length;
  return index === next || index === prev;
}

/**
 * Preload budget for a stacked clip. A detached clip has no `src` to fetch, so
 * it never asks for data — including while it is the active clip and the
 * playlist is paused (out of view or reduced motion).
 */
export function peakVideoPreload(
  isActive: boolean,
  attachSrc: boolean,
): "auto" | "metadata" | "none" {
  if (!attachSrc) return "none";
  return isActive ? "auto" : "metadata";
}

/**
 * A clip that loses its `src` keeps showing the last decoded frame, so the
 * poster only comes back when the element is reloaded on that transition.
 */
export function shouldRestorePeakPoster(
  wasAttached: boolean,
  isAttached: boolean,
): boolean {
  return wasAttached && !isAttached;
}

/**
 * Clamps a possibly stale active index onto the current playlist so a shrunk
 * list falls back to the first clip instead of leaving nothing active.
 */
export function resolvePeakActiveIndex(
  activeIndex: number,
  length: number,
): number {
  return activeIndex >= 0 && activeIndex < length ? activeIndex : 0;
}

/**
 * Finds the next playable clip, wrapping around the playlist. If the current
 * clip is still playable and every other clip has failed, it is returned so a
 * single surviving clip can continue looping. A failed current clip returns
 * null when no playable clips remain.
 */
export function nextPeakPlayableIndex(
  ids: readonly string[],
  activeIndex: number,
  failedIds: ReadonlySet<string>,
): number | null {
  if (ids.length === 0) return null;

  const resolvedActiveIndex = resolvePeakActiveIndex(activeIndex, ids.length);
  for (let offset = 1; offset < ids.length; offset += 1) {
    const index = (resolvedActiveIndex + offset) % ids.length;
    if (!failedIds.has(ids[index])) return index;
  }

  return failedIds.has(ids[resolvedActiveIndex]!) ? null : resolvedActiveIndex;
}

export type PeakClipState = {
  readonly id: string;
  readonly index: number;
  readonly isActive: boolean;
  readonly attachSrc: boolean;
};

/**
 * The playlist's whole per-render decision: which clip is active and which
 * clips carry a `src`. Rendering and the playback effect both read this so the
 * two can never disagree about the active clip.
 */
export function peakClipStates({
  ids,
  activeIndex,
  hasBeenVisible,
  inView,
  reducedMotion,
  failedIds,
}: {
  ids: readonly string[];
  activeIndex: number;
  hasBeenVisible: boolean;
  inView: boolean;
  reducedMotion: boolean;
  failedIds: ReadonlySet<string>;
}): PeakClipState[] {
  const length = ids.length;
  const resolvedActiveIndex = resolvePeakActiveIndex(activeIndex, length);
  const hasPlaylist = length > 1;

  return ids.map((id, index) => {
    const isActive = index === resolvedActiveIndex;
    const attachSrc =
      !reducedMotion &&
      !failedIds.has(id) &&
      shouldAttachPeakVideoSrc({
        hasBeenVisible,
        inView,
        isActive,
        hasPlaylist,
        index,
        activeIndex: resolvedActiveIndex,
        length,
      });
    return { id, index, isActive, attachSrc };
  });
}
