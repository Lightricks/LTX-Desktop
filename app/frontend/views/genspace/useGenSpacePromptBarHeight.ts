import { useCallback, useRef, useState } from 'react'
import type { PanelImperativeHandle } from 'react-resizable-panels'
import {
  DEFAULT_PROMPT_BAR_HEIGHT,
  PROMPT_BAR_HEIGHT_LIMITS,
  clampPromptBarHeight,
  loadPromptBarHeight,
  savePromptBarHeight,
} from '../../lib/genspace-layout'

export function useGenSpacePromptBarHeight() {
  const initialHeightRef = useRef(loadPromptBarHeight())
  const heightRef = useRef(initialHeightRef.current)
  const panelRef = useRef<PanelImperativeHandle | null>(null)
  const [promptBarHeight, setPromptBarHeight] = useState(initialHeightRef.current)

  const persistHeight = useCallback((inPixels: number) => {
    const next = clampPromptBarHeight(inPixels)
    if (next === heightRef.current) return
    heightRef.current = next
    setPromptBarHeight(next)
    savePromptBarHeight(next)
  }, [])

  const resetHeight = useCallback(() => {
    heightRef.current = DEFAULT_PROMPT_BAR_HEIGHT
    setPromptBarHeight(DEFAULT_PROMPT_BAR_HEIGHT)
    savePromptBarHeight(DEFAULT_PROMPT_BAR_HEIGHT)
    panelRef.current?.resize(DEFAULT_PROMPT_BAR_HEIGHT)
  }, [])

  return {
    initialHeight: initialHeightRef.current,
    promptBarHeight,
    panelRef,
    persistHeight,
    resetHeight,
    limits: PROMPT_BAR_HEIGHT_LIMITS,
  }
}
