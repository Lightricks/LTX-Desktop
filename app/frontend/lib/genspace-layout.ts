export const GENSPACE_LAYOUT_STORAGE_KEY = 'ltx-genspace-layout'

export const DEFAULT_PROMPT_BAR_HEIGHT = 160

export const PROMPT_BAR_HEIGHT_LIMITS = { min: 140, max: 400 } as const

/** Gap (px) between the prompt bar and the floating LoRA/IC-LoRA chip overlay. */
export const LORA_CHIP_OVERLAY_GAP = 4

/**
 * One-row SelectedLoraInfo estimate: py-2 (16) + 11px text at 1.5 leading (~17)
 * + 2px border + mb-2 (8). Used as a floor until the overlay is measured —
 * wrapping names and FreeApiKeyBubble stacking can be taller.
 */
export const LORA_CHIP_OVERLAY_HEIGHT = 43

/** Default bottom inset for mode panels (`pb-4`). */
const DEFAULT_MODE_PANEL_PADDING_BOTTOM = 16

export type LayoutStorage = Pick<Storage, 'getItem' | 'setItem'>

/**
 * Bottom padding for non-gallery GenSpace panels. Pass the overlay's measured
 * height so clearance tracks wrap/stack; `measuredOverlayHeightPx` of 0 keeps
 * the one-row floor so chrome is not uncovered on the first layout pass.
 */
export function genspaceModePanelPaddingBottom(
  overlayVisible: boolean,
  measuredOverlayHeightPx = 0,
): number {
  if (!overlayVisible) return DEFAULT_MODE_PANEL_PADDING_BOTTOM
  return Math.max(LORA_CHIP_OVERLAY_HEIGHT, measuredOverlayHeightPx)
}

export function clampPromptBarHeight(height: number): number {
  return Math.max(
    PROMPT_BAR_HEIGHT_LIMITS.min,
    Math.min(PROMPT_BAR_HEIGHT_LIMITS.max, Math.round(height)),
  )
}

export function loadPromptBarHeight(storage: LayoutStorage = globalThis.localStorage): number {
  try {
    const stored = storage?.getItem(GENSPACE_LAYOUT_STORAGE_KEY)
    if (!stored) return DEFAULT_PROMPT_BAR_HEIGHT
    const parsed = JSON.parse(stored) as { promptBarHeight?: unknown }
    if (typeof parsed.promptBarHeight !== 'number' || !Number.isFinite(parsed.promptBarHeight)) {
      return DEFAULT_PROMPT_BAR_HEIGHT
    }
    return clampPromptBarHeight(parsed.promptBarHeight)
  } catch {
    return DEFAULT_PROMPT_BAR_HEIGHT
  }
}

export function savePromptBarHeight(height: number, storage: LayoutStorage = globalThis.localStorage): void {
  try {
    storage?.setItem(
      GENSPACE_LAYOUT_STORAGE_KEY,
      JSON.stringify({ promptBarHeight: clampPromptBarHeight(height) }),
    )
  } catch {
    // Private mode / quota — layout just won't persist.
  }
}
