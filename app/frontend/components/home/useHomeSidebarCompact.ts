import { useEffect, useState } from 'react'
import {
  HOME_SIDEBAR_COMPACT_QUERY,
  matchesHomeSidebarCompact,
  type CompactMediaWindow,
} from './home-sidebar-expansion'

function currentWindow(): CompactMediaWindow | undefined {
  return typeof window === 'undefined' ? undefined : window
}

/** Live compact-sidebar viewport, seeded synchronously on mount. */
export function useHomeSidebarCompact(): boolean {
  const [compact, setCompact] = useState(() =>
    matchesHomeSidebarCompact(currentWindow()),
  )

  useEffect(() => {
    const win = currentWindow()
    if (!win || typeof win.matchMedia !== 'function') return
    const mediaQuery = win.matchMedia(HOME_SIDEBAR_COMPACT_QUERY)
    const update = () => setCompact(mediaQuery.matches)
    update()
    mediaQuery.addEventListener('change', update)
    return () => mediaQuery.removeEventListener('change', update)
  }, [])

  return compact
}
