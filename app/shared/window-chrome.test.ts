import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { macTrafficLightsOnEnd } from './window-chrome.ts'

describe('macTrafficLightsOnEnd', () => {
  it('detects RTL on Mac from the locale script, not the bare language code (he, fa, and an explicit ff-Adlm)', () => {
    for (const locale of ['he', 'he-IL', 'ar', 'fa', 'ur', 'ff-Adlm']) {
      assert.equal(macTrafficLightsOnEnd('darwin', locale), true, locale)
    }
    for (const locale of ['en', 'en-US']) {
      assert.equal(macTrafficLightsOnEnd('darwin', locale), false, locale)
    }
  })

  it('is false on Windows and Linux for an RTL locale (their native caption does not mirror)', () => {
    assert.equal(macTrafficLightsOnEnd('win32', 'he'), false)
    assert.equal(macTrafficLightsOnEnd('linux', 'ar'), false)
    assert.equal(macTrafficLightsOnEnd(undefined, 'he'), false)
  })

  it('falls back to false for a malformed locale tag instead of throwing and blanking the app', () => {
    assert.equal(macTrafficLightsOnEnd('darwin', 'xx_'), false)
  })
})
