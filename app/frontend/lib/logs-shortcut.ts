export interface LogsKeyInput {
  platform: string | undefined
  /** `KeyboardEvent.code`: the physical key. It matches the L key position, not the printed letter. */
  code: string
  metaKey: boolean
  ctrlKey: boolean
  shiftKey: boolean
  altKey: boolean
}

/** Cmd+Shift+L on macOS, Ctrl+Shift+L on Windows and Linux. */
export function isLogsShortcut(input: LogsKeyInput): boolean {
  if (input.code !== 'KeyL' || !input.shiftKey || input.altKey) return false
  return input.platform === 'darwin'
    ? input.metaKey && !input.ctrlKey
    : input.ctrlKey && !input.metaKey
}

export function logsShortcutLabel(platform: string | undefined): string {
  return platform === 'darwin' ? '⌘⇧L' : 'Ctrl+Shift+L'
}
