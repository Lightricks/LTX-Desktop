import { useEffect } from 'react'

/**
 * Calls `onClose` when the user presses Escape.
 * An Escape that another control already used does nothing. For example, Escape
 * in the open font size menu closes only that menu. The same holds for Escape
 * while an input method (IME) is composing text.
 */
export function useCloseOnEscape(onClose: () => void, enabled: boolean) {
  useEffect(() => {
    if (!enabled) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || event.defaultPrevented || event.isComposing) return
      event.preventDefault()
      onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [enabled, onClose])
}
