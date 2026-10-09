const MAX_LABEL = 80

/** Short label for the Connected devices list. The API already summarizes UAs. */
export function pairedDeviceDisplayName(name: string): string {
  const trimmed = name.trim()
  if (trimmed.length === 0) return "Paired device"
  return trimmed.slice(0, MAX_LABEL)
}
