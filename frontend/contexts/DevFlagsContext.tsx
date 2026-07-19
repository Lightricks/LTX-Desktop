import React, { createContext, useContext, useState, useEffect, useCallback } from 'react'

// Dev / escape-hatch feature flags. Client-only, per-machine (localStorage) —
// not a user setting and not synced. Toggle via the Dev Panel (Ctrl/Cmd+Shift+D).
// Adding a feature flag = one entry in DEV_FLAGS; read it from a single gate where
// the feature is used (avoid sprinkling the flag across call sites).
export type DevFlagKey = 'customIcLora' | 'advancedIcLoraControls'

interface DevFlagSpec {
  key: DevFlagKey
  label: string
  description: string
}

export const DEV_FLAGS: DevFlagSpec[] = [
  {
    key: 'customIcLora',
    label: 'Custom IC-LoRA',
    description: 'Show the "Custom IC-LoRA" conditioning option (user-supplied weights + control video). Off by default — local results are currently low quality. Built-in Canny/Depth IC-LoRA is unaffected.',
  },
  {
    key: 'advancedIcLoraControls',
    label: 'Advanced IC-LoRA controls',
    description: 'Expose all IC-LoRA settings (skip stage 2, resolution factor, audio, strengths) for catalog and custom modes. Off by default — settings use the catalog IC-LoRA defaults.',
  },
]

type Flags = Record<DevFlagKey, boolean>

const DEFAULT_FLAGS: Flags = { customIcLora: false, advancedIcLoraControls: false }
const STORAGE_KEY = 'ltx-dev-flags'

function loadFlags(): Flags {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return { ...DEFAULT_FLAGS, ...(JSON.parse(raw) as Partial<Flags>) }
  } catch { /* ignore */ }
  return { ...DEFAULT_FLAGS }
}

interface DevFlagsState {
  flags: Flags
  setFlag: (key: DevFlagKey, on: boolean) => void
  isPanelOpen: boolean
  setPanelOpen: (open: boolean) => void
}

const DevFlagsContext = createContext<DevFlagsState | null>(null)

export function DevFlagsProvider({ children }: { children: React.ReactNode }) {
  const [flags, setFlags] = useState<Flags>(() => loadFlags())
  const [isPanelOpen, setPanelOpen] = useState(false)

  const setFlag = useCallback((key: DevFlagKey, on: boolean) => {
    setFlags(prev => {
      const next = { ...prev, [key]: on }
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(next)) } catch { /* ignore */ }
      return next
    })
  }, [])

  // Global toggle: Ctrl/Cmd + Shift + D. Capture phase so component-level keydown
  // handlers don't swallow it; preventDefault avoids any Electron/Chromium default.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === 'd') {
        e.preventDefault()
        e.stopPropagation()
        setPanelOpen(prev => !prev)
      }
    }
    window.addEventListener('keydown', onKey, { capture: true })
    return () => window.removeEventListener('keydown', onKey, { capture: true })
  }, [])

  return (
    <DevFlagsContext.Provider value={{ flags, setFlag, isPanelOpen, setPanelOpen }}>
      {children}
    </DevFlagsContext.Provider>
  )
}

export function useDevFlags(): DevFlagsState {
  const ctx = useContext(DevFlagsContext)
  if (!ctx) throw new Error('useDevFlags must be used within DevFlagsProvider')
  return ctx
}
