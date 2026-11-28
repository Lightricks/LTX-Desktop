import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react'
import {
  KeyboardLayout,
  KeyboardPreset,
  BUILT_IN_PRESETS,
  LTX_DEFAULT_LAYOUT,
  cloneLayout,
  isReservedCombo,
  ActionId,
} from '../lib/keyboard-shortcuts'

interface KeyboardShortcutsState {
  // Current active layout
  activeLayout: KeyboardLayout
  activePresetId: string
  // All presets (built-in + user)
  presets: KeyboardPreset[]
  // Actions
  switchPreset: (presetId: string) => void
  updateBinding: (actionId: ActionId, combos: KeyboardLayout[ActionId]) => void
  resetToPreset: (presetId: string) => void
  saveAsCustomPreset: (name: string) => void
  deleteCustomPreset: (presetId: string) => void
  // Whether the shortcuts editor modal is open
  isEditorOpen: boolean
  setEditorOpen: (open: boolean) => void
}

const KeyboardShortcutsContext = createContext<KeyboardShortcutsState | null>(null)

const STORAGE_KEY = 'ltx-keyboard-shortcuts'

interface PersistedState {
  activePresetId: string
  customLayout?: KeyboardLayout   // Only stored when user has modified bindings
  customPresets?: KeyboardPreset[]
}

function loadFromStorage(): PersistedState | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return JSON.parse(raw)
  } catch { /* ignore */ }
  return null
}

function saveToStorage(state: PersistedState) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  } catch { /* ignore */ }
}

export function KeyboardShortcutsProvider({ children }: { children: React.ReactNode }) {
  const stored = useRef(loadFromStorage())

  const [activePresetId, setActivePresetId] = useState<string>(stored.current?.activePresetId || 'ltx-default')
  const [customLayout, setCustomLayout] = useState<KeyboardLayout | null>(stored.current?.customLayout || null)
  const [customPresets, setCustomPresets] = useState<KeyboardPreset[]>(stored.current?.customPresets || [])
  const [isEditorOpen, setEditorOpen] = useState(false)

  // Resolve active layout: if customLayout is set, use it; otherwise use the preset's layout
  const activeLayout: KeyboardLayout = customLayout
    || [...BUILT_IN_PRESETS, ...customPresets].find(p => p.id === activePresetId)?.layout
    || LTX_DEFAULT_LAYOUT

  // Keep a ref to the active layout so updateBinding always reads the latest
  const activeLayoutRef = useRef(activeLayout)
  activeLayoutRef.current = activeLayout

  const allPresets = [...BUILT_IN_PRESETS, ...customPresets]

  // Persist whenever state changes
  useEffect(() => {
    saveToStorage({
      activePresetId,
      customLayout: customLayout || undefined,
      customPresets: customPresets.length > 0 ? customPresets : undefined,
    })
  }, [activePresetId, customLayout, customPresets])

  const switchPreset = useCallback((presetId: string) => {
    setActivePresetId(presetId)
    setCustomLayout(null) // Clear any custom modifications — use the preset directly
  }, [])

  const updateBinding = useCallback((actionId: ActionId, combos: KeyboardLayout[ActionId]) => {
    // Always read the LATEST layout from the ref to avoid stale closure issues
    setCustomLayout(prev => {
      const base = prev || cloneLayout(activeLayoutRef.current)
      const updated = { ...base, [actionId]: combos?.filter((combo) => !isReservedCombo(combo)) }
      return updated
    })
  }, []) // No deps needed — reads from ref

  const resetToPreset = useCallback((presetId: string) => {
    setActivePresetId(presetId)
    setCustomLayout(null)
  }, [])

  const saveAsCustomPreset = useCallback((name: string) => {
    const preset: KeyboardPreset = {
      id: `custom-${Date.now()}`,
      name,
      description: 'Custom keyboard layout',
      layout: cloneLayout(activeLayoutRef.current),
      builtIn: false,
    }
    setCustomPresets(prev => [...prev, preset])
    setActivePresetId(preset.id)
    setCustomLayout(null)
  }, [])

  const deleteCustomPreset = useCallback((presetId: string) => {
    setCustomPresets(prev => prev.filter(p => p.id !== presetId))
    if (activePresetId === presetId) {
      setActivePresetId('ltx-default')
      setCustomLayout(null)
    }
  }, [activePresetId])

  return (
    <KeyboardShortcutsContext.Provider value={{
      activeLayout,
      activePresetId,
      presets: allPresets,
      switchPreset,
      updateBinding,
      resetToPreset,
      saveAsCustomPreset,
      deleteCustomPreset,
      isEditorOpen,
      setEditorOpen,
    }}>
      {children}
    </KeyboardShortcutsContext.Provider>
  )
}

export function useKeyboardShortcuts() {
  const ctx = useContext(KeyboardShortcutsContext)
  if (!ctx) throw new Error('useKeyboardShortcuts must be used inside KeyboardShortcutsProvider')
  return ctx
}
