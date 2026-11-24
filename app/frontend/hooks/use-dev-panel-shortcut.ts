import { useEffect } from 'react'

// Global Dev Panel toggle: Ctrl/Cmd + Shift + D, shared by Desktop and the remote
// browser. Capture phase so component-level keydown handlers don't swallow it;
// preventDefault avoids any Electron/Chromium default.
//
// Browsers: Ctrl+Shift+D works everywhere (verified in a Mac browser). Cmd+Shift+D is
// likely claimed by the browser itself on Mac (Chrome/Firefox "bookmark all tabs",
// Safari "add to reading list") and may never reach the page. In the remote, Ctrl is
// the dependable choice, with `?devpanel=1` as the fallback.
export function useDevPanelShortcut(onToggle: () => void): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === 'd') {
        e.preventDefault()
        e.stopPropagation()
        onToggle()
      }
    }
    window.addEventListener('keydown', onKey, { capture: true })
    return () => window.removeEventListener('keydown', onKey, { capture: true })
  }, [onToggle])
}
