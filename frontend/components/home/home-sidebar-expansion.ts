export const HOME_SIDEBAR_EXPANDED_WIDTH_PX = 240
export const HOME_SIDEBAR_COLLAPSE_DELAY_MS = 150
/** Keep in sync with `@media (max-width: 980px)` in home layout / sidebar CSS. */
export const HOME_SIDEBAR_COMPACT_MAX_WIDTH_PX = 980
export const HOME_SIDEBAR_COMPACT_QUERY = `(max-width: ${HOME_SIDEBAR_COMPACT_MAX_WIDTH_PX}px)`

/**
 * Vocabulary (used consistently across state, helpers, and labels):
 * - `pinned`: the sidebar is docked open (user intent, persists while mounted).
 * - `open`: the sidebar is currently visible — pinned, or transiently shown
 *   via hover / keyboard focus / an open menu.
 */
export type HomeSidebarExpansionState = {
  pinned: boolean
  hovered: boolean
  focused: boolean
  menuOpen: boolean
}

export function isHomeSidebarOpen(input: HomeSidebarExpansionState): boolean {
  return input.pinned || input.menuOpen || input.hovered || input.focused
}

export type CompactMediaQuery = {
  matches: boolean
  addEventListener(type: 'change', listener: () => void): void
  removeEventListener(type: 'change', listener: () => void): void
}

export type CompactMediaWindow = {
  matchMedia?: (query: string) => CompactMediaQuery
}

export function matchesHomeSidebarCompact(
  win: CompactMediaWindow | undefined,
): boolean {
  if (!win || typeof win.matchMedia !== 'function') return false
  return win.matchMedia(HOME_SIDEBAR_COMPACT_QUERY).matches
}

export function homeSidebarContentOffsetPx(pinned: boolean): number {
  return pinned ? HOME_SIDEBAR_EXPANDED_WIDTH_PX : 0
}

export function homeSidebarPinLabel(pinned: boolean): string {
  return pinned ? 'Unpin sidebar' : 'Pin sidebar'
}

/** Collapsing drops hover/focus so the overlay can hide while the pointer stays over it. */
export function applyHomeSidebarPinChange(
  state: HomeSidebarExpansionState,
  nextPinned: boolean,
): HomeSidebarExpansionState {
  if (nextPinned) {
    return { ...state, pinned: true }
  }
  return {
    ...state,
    pinned: false,
    hovered: false,
    focused: false,
  }
}

export function createHomeSidebarCollapseDelay<TimeoutId>(deps: {
  delayMs: number
  setTimeoutFn: (fn: () => void, ms: number) => TimeoutId
  clearTimeoutFn: (id: TimeoutId) => void
}) {
  let pending: TimeoutId | undefined

  return {
    schedule(callback: () => void) {
      if (pending !== undefined) deps.clearTimeoutFn(pending)
      pending = deps.setTimeoutFn(() => {
        pending = undefined
        callback()
      }, deps.delayMs)
    },
    cancel() {
      if (pending === undefined) return
      deps.clearTimeoutFn(pending)
      pending = undefined
    },
  }
}
