import { useEffect, useState } from 'react'

export const REDUCED_MOTION_QUERY = '(prefers-reduced-motion: reduce)'

export type ReducedMotionMediaQuery = {
  matches: boolean
  addEventListener(type: 'change', listener: () => void): void
  removeEventListener(type: 'change', listener: () => void): void
}

export type ReducedMotionWindow = {
  matchMedia?: (query: string) => ReducedMotionMediaQuery
}

/**
 * Reads the reduced-motion preference. Environments without `matchMedia`
 * (tests, SSR) report motion as allowed, matching the browser default.
 */
export function matchesReducedMotion(win: ReducedMotionWindow | undefined): boolean {
  if (!win || typeof win.matchMedia !== 'function') return false
  return win.matchMedia(REDUCED_MOTION_QUERY).matches
}

function currentWindow(): ReducedMotionWindow | undefined {
  return typeof window === 'undefined' ? undefined : window
}

/** Live `prefers-reduced-motion: reduce` state, seeded synchronously on mount. */
export function usePrefersReducedMotion(): boolean {
  const [reducedMotion, setReducedMotion] = useState(() =>
    matchesReducedMotion(currentWindow()),
  )

  useEffect(() => {
    const win = currentWindow()
    if (!win || typeof win.matchMedia !== 'function') return
    const mediaQuery = win.matchMedia(REDUCED_MOTION_QUERY)
    const update = () => setReducedMotion(mediaQuery.matches)
    update()
    mediaQuery.addEventListener('change', update)
    return () => mediaQuery.removeEventListener('change', update)
  }, [])

  return reducedMotion
}
