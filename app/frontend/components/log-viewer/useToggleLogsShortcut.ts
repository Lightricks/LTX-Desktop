import { useEffect } from 'react'
import { isLogsShortcut } from '../../lib/logs-shortcut'

/**
 * Calls `onToggle` when the user presses the logs shortcut, from anywhere in the app.
 * Pass `enabled: false` while another part of the app records keystrokes.
 */
export function useToggleLogsShortcut(onToggle: () => void, enabled: boolean) {
  useEffect(() => {
    if (!enabled) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return
      const { code, metaKey, ctrlKey, shiftKey, altKey } = event
      const platform = document.documentElement.dataset.platform
      if (!isLogsShortcut({ platform, code, metaKey, ctrlKey, shiftKey, altKey })) return
      event.preventDefault()
      event.stopPropagation()
      // A held key repeats keydown. Ignore it so the dialog does not flicker.
      if (!event.repeat) onToggle()
    }
    window.addEventListener('keydown', onKeyDown, { capture: true })
    return () => window.removeEventListener('keydown', onKeyDown, { capture: true })
  }, [enabled, onToggle])
}
