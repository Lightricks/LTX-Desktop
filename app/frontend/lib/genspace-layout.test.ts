import assert from 'node:assert/strict'
import { beforeEach, describe, it } from 'node:test'
import {
  DEFAULT_PROMPT_BAR_HEIGHT,
  GENSPACE_LAYOUT_STORAGE_KEY,
  LORA_CHIP_OVERLAY_HEIGHT,
  PROMPT_BAR_HEIGHT_LIMITS,
  genspaceModePanelPaddingBottom,
  loadPromptBarHeight,
  savePromptBarHeight,
} from './genspace-layout.ts'

function createMemoryStorage(initial: Record<string, string> = {}) {
  const data = new Map(Object.entries(initial))
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value)
    },
  }
}

describe('genspace prompt-bar layout', () => {
  let storage: ReturnType<typeof createMemoryStorage>

  beforeEach(() => {
    storage = createMemoryStorage()
  })

  it('loads the default height when nothing is stored', () => {
    assert.equal(loadPromptBarHeight(storage), DEFAULT_PROMPT_BAR_HEIGHT)
  })

  it('round-trips a saved height', () => {
    savePromptBarHeight(240, storage)
    assert.equal(loadPromptBarHeight(storage), 240)
    assert.equal(
      storage.getItem(GENSPACE_LAYOUT_STORAGE_KEY),
      JSON.stringify({ promptBarHeight: 240 }),
    )
  })

  it('clamps heights outside the allowed range', () => {
    savePromptBarHeight(PROMPT_BAR_HEIGHT_LIMITS.min - 50, storage)
    assert.equal(loadPromptBarHeight(storage), PROMPT_BAR_HEIGHT_LIMITS.min)

    savePromptBarHeight(PROMPT_BAR_HEIGHT_LIMITS.max + 80, storage)
    assert.equal(loadPromptBarHeight(storage), PROMPT_BAR_HEIGHT_LIMITS.max)
  })

  it('falls back to the default when the stored height is missing, not a number, or invalid JSON', () => {
    storage.setItem(GENSPACE_LAYOUT_STORAGE_KEY, '{not-json')
    assert.equal(loadPromptBarHeight(storage), DEFAULT_PROMPT_BAR_HEIGHT)

    storage.setItem(GENSPACE_LAYOUT_STORAGE_KEY, JSON.stringify({}))
    assert.equal(loadPromptBarHeight(storage), DEFAULT_PROMPT_BAR_HEIGHT)

    storage.setItem(GENSPACE_LAYOUT_STORAGE_KEY, JSON.stringify({ promptBarHeight: 'tall' }))
    assert.equal(loadPromptBarHeight(storage), DEFAULT_PROMPT_BAR_HEIGHT)
  })
})

describe('genspace LoRA-chip overlay clearance', () => {
  it('reserves the measured overlay height and never less than one SelectedLoraInfo row', () => {
    assert.equal(genspaceModePanelPaddingBottom(false, 80), 16)
    assert.equal(genspaceModePanelPaddingBottom(true, 0), LORA_CHIP_OVERLAY_HEIGHT)
    assert.equal(genspaceModePanelPaddingBottom(true, 80), 80)

    // SelectedLoraInfo: py-2 (16) + 11px text at 1.5 leading (~17) + 2px border + mb-2 (8).
    const selectedLoraInfoMinHeight = 16 + 17 + 2 + 8
    assert.ok(LORA_CHIP_OVERLAY_HEIGHT >= selectedLoraInfoMinHeight)
    assert.ok(genspaceModePanelPaddingBottom(true, 20) >= selectedLoraInfoMinHeight)
  })
})
