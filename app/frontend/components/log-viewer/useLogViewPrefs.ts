import { useState } from 'react'

export const LOG_FONT_SIZES = [
  { value: 'small', label: 'Small', px: 9 },
  { value: 'medium', label: 'Medium', px: 11 },
  { value: 'large', label: 'Large', px: 13 },
  { value: 'xlarge', label: 'Extra large', px: 15 },
] as const

export type LogFontSizeId = (typeof LOG_FONT_SIZES)[number]['value']

interface LogViewPrefs {
  fontSize: LogFontSizeId
  wrap: boolean
}

const DEFAULT_PREFS: LogViewPrefs = { fontSize: 'medium', wrap: true }
const STORAGE_KEY = 'ltx.logViewer.prefs'

function parseLogFontSize(value: unknown): LogFontSizeId {
  return LOG_FONT_SIZES.find((size) => size.value === value)?.value ?? DEFAULT_PREFS.fontSize
}

function readStoredPrefs(): LogViewPrefs {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null')
    if (typeof stored !== 'object' || stored === null) return DEFAULT_PREFS
    const { fontSize, wrap } = stored as Record<string, unknown>
    return {
      fontSize: parseLogFontSize(fontSize),
      wrap: typeof wrap === 'boolean' ? wrap : DEFAULT_PREFS.wrap,
    }
  } catch {
    return DEFAULT_PREFS
  }
}

/** Log text size and word wrap, remembered across sessions. */
export function useLogViewPrefs() {
  const [prefs, setPrefs] = useState<LogViewPrefs>(readStoredPrefs)

  const update = (patch: Partial<LogViewPrefs>) => {
    const next = { ...prefs, ...patch }
    setPrefs(next)
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
    } catch {
      // Storage can be full or blocked. The prefs then last for this session only.
    }
  }

  return {
    fontSize: prefs.fontSize,
    fontSizePx: LOG_FONT_SIZES.find((size) => size.value === prefs.fontSize)?.px ?? 11,
    setFontSize: (value: string) => update({ fontSize: parseLogFontSize(value) }),
    wrap: prefs.wrap,
    setWrap: (wrap: boolean) => update({ wrap }),
  }
}
