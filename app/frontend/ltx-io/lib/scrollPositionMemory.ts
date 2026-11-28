const STORAGE_PREFIX = "ltx-scroll-top:";

export type ScrollStorage = Pick<Storage, "getItem" | "setItem">;

export function scrollStorageKey(pageId: string): string {
  return STORAGE_PREFIX + pageId;
}

/** The session storage of this window, or null when the browser blocks it. */
export function getSessionScrollStorage(): ScrollStorage | null {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

/** A position worth restoring: a finite number above 0. Anything else opens at the top. */
function parseRestorableScrollTop(raw: string | null): number | null {
  if (raw == null) return null;
  const value = Number(raw);
  return Number.isFinite(value) && value > 0 ? value : null;
}

/**
 * The save and restore rules for the scroll position of a page, without React or the DOM.
 *
 * `begin` is the restore step of a page. A page is saved only after its `begin`. Without that
 * rule, the scroll event of an empty or changing page saves 0 over the real position.
 * Storage errors (blocked storage, full quota) are not fatal: the page opens at the top.
 */
export function createScrollPositionMemory(storage: ScrollStorage | null) {
  let trackedPageId: string | null = null;

  return {
    isTracking(pageId: string): boolean {
      return trackedPageId === pageId;
    },

    /**
     * Starts the tracking of a page. Returns the position to apply, or null when there is
     * nothing to restore (no saved value, an invalid value, 0, or `restore` is false).
     */
    begin(pageId: string, restore: boolean): number | null {
      trackedPageId = pageId;
      if (!restore || storage == null) return null;
      try {
        return parseRestorableScrollTop(storage.getItem(scrollStorageKey(pageId)));
      } catch {
        return null;
      }
    },

    /** Saves the position, only for the page that `begin` started. */
    save(pageId: string, top: number): void {
      if (trackedPageId !== pageId || storage == null) return;
      try {
        storage.setItem(scrollStorageKey(pageId), String(Math.round(top)));
      } catch {
        // Losing the position is harmless.
      }
    },
  };
}
