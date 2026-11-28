import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import {
  parseSettingsInitialReason,
  parseSettingsScrollAnchor,
  parseSettingsTabId,
} from './settings-navigation.ts'

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
