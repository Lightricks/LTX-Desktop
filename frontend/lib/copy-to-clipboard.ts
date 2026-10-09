/**
 * Copy text to the clipboard. The async Clipboard API only exists in secure
 * contexts, and the remote app is served over plain http on the LAN, so fall
 * back to a hidden textarea and `execCommand('copy')`.
 */
export async function copyTextToClipboard(value: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(value)
    return true
  } catch {
    const previouslyFocused = document.activeElement as HTMLElement | null
    try {
      const textarea = document.createElement('textarea')
      textarea.value = value
      textarea.setAttribute('readonly', '')
      textarea.style.position = 'fixed'
      textarea.style.left = '-9999px'
      document.body.appendChild(textarea)
      textarea.select()
      textarea.setSelectionRange(0, value.length) // iOS ignores select() alone
      const copied = document.execCommand('copy')
      document.body.removeChild(textarea)
      // select() moved focus to the textarea; give it back (keyboard users)
      previouslyFocused?.focus?.()
      return copied
    } catch {
      return false
    }
  }
}
