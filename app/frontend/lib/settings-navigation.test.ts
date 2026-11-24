import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import {
  isSettingsTabAvailable,
  parseSettingsInitialReason,
  parseSettingsScrollAnchor,
  parseSettingsTabId,
} from './settings-navigation.ts'

describe('isSettingsTabAvailable', () => {
  it('shows every tab when local generation is available', () => {
    for (const tab of ['general', 'models', 'apiKeys', 'legacy', 'about'] as const) {
      assert.equal(isSettingsTabAvailable(tab, { forceApiGenerations: false }), true)
    }
  })

  it('hides only Models in force-API mode', () => {
    assert.equal(isSettingsTabAvailable('models', { forceApiGenerations: true }), false)
    for (const tab of ['general', 'apiKeys', 'legacy', 'about'] as const) {
      assert.equal(isSettingsTabAvailable(tab, { forceApiGenerations: true }), true)
    }
  })

  it('keeps Legacy reachable in force-API mode: it owns the assets path and the seed lock', () => {
    // Regression: Legacy was listed in the tab strip but a separate guard bounced the
    // selection back to General, so the tab could be seen and never opened.
    assert.equal(isSettingsTabAvailable('legacy', { forceApiGenerations: true }), true)
  })
})

describe('parseSettingsTabId', () => {
  it('accepts the real tab ids', () => {
    assert.equal(parseSettingsTabId('models'), 'models')
    assert.equal(parseSettingsTabId('legacy'), 'legacy')
  })

  it('maps the legacy promptEnhancer hash onto the tab that now owns the row', () => {
    assert.equal(parseSettingsTabId('promptEnhancer'), 'general')
  })

  it('rejects unknown values and null', () => {
    assert.equal(parseSettingsTabId('nope'), undefined)
    assert.equal(parseSettingsTabId(null), undefined)
  })
})

describe('parseSettingsScrollAnchor', () => {
  it('keeps promptEnhancer as a row anchor, not just a tab', () => {
    assert.equal(parseSettingsScrollAnchor('promptEnhancer'), 'promptEnhancer')
  })

  it('keeps textEncoding as a row anchor on General', () => {
    assert.equal(parseSettingsScrollAnchor('textEncoding'), 'textEncoding')
    assert.equal(parseSettingsTabId('textEncoding'), 'general')
  })

  it('is undefined for tab ids and unknown values', () => {
    assert.equal(parseSettingsScrollAnchor('general'), undefined)
    assert.equal(parseSettingsScrollAnchor('nope'), undefined)
    assert.equal(parseSettingsScrollAnchor(null), undefined)
  })
})

describe('parseSettingsInitialReason', () => {
  it('accepts the reasons that focus an API key field', () => {
    assert.equal(parseSettingsInitialReason('geminiKeyRequired'), 'geminiKeyRequired')
    assert.equal(parseSettingsInitialReason('ltxKeyRequired'), 'ltxKeyRequired')
    assert.equal(parseSettingsInitialReason('other'), undefined)
    assert.equal(parseSettingsInitialReason(null), undefined)
  })
})
