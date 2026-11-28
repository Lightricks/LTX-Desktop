import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  GEMINI_KEY_REQUIRED_SETTINGS_DETAIL,
  isEnhanceBlockedByGeneration,
  isEnhanceBlockedByMissingGeminiKey,
} from './enhance-gemini-key.ts'

const blocked = {
  enhanceAvailableForMode: true,
  enhanceProvider: 'api' as const,
  hasGeminiApiKey: false,
  hasEnhanceInput: true,
}

describe('isEnhanceBlockedByMissingGeminiKey', () => {
  it('is true when Enhance (API) is selected and no Gemini key is configured', () => {
    assert.equal(isEnhanceBlockedByMissingGeminiKey(blocked), true)
  })

  it('is false when enhance is not available for the current mode', () => {
    assert.equal(
      isEnhanceBlockedByMissingGeminiKey({ ...blocked, enhanceAvailableForMode: false }),
      false,
    )
  })

  it('is false when a Gemini key is already configured', () => {
    assert.equal(
      isEnhanceBlockedByMissingGeminiKey({ ...blocked, hasGeminiApiKey: true }),
      false,
    )
  })

  it('is false when Enhance is using the local provider, even without a Gemini key', () => {
    assert.equal(
      isEnhanceBlockedByMissingGeminiKey({ ...blocked, enhanceProvider: 'local' }),
      false,
    )
  })

  it('is false when there is no prompt or image to enhance', () => {
    assert.equal(
      isEnhanceBlockedByMissingGeminiKey({ ...blocked, hasEnhanceInput: false }),
      false,
    )
  })
})

describe('isEnhanceBlockedByGeneration', () => {
  it('blocks local Enhance while this project or another project generates', () => {
    for (const running of [
      { isGenerationInProgressForEnhance: true, isOtherGenerationRunning: false },
      { isGenerationInProgressForEnhance: false, isOtherGenerationRunning: true },
    ]) {
      assert.equal(isEnhanceBlockedByGeneration({ enhanceProvider: 'local', ...running }), true)
    }
  })

  it('never blocks Gemini Enhance, which runs beside a generation', () => {
    assert.equal(
      isEnhanceBlockedByGeneration({
        enhanceProvider: 'api',
        isGenerationInProgressForEnhance: true,
        isOtherGenerationRunning: true,
      }),
      false,
    )
  })

  it('does not block local Enhance when nothing is generating', () => {
    assert.equal(
      isEnhanceBlockedByGeneration({
        enhanceProvider: 'local',
        isGenerationInProgressForEnhance: false,
        isOtherGenerationRunning: false,
      }),
      false,
    )
  })
})

describe('GEMINI_KEY_REQUIRED_SETTINGS_DETAIL', () => {
  it('opens Settings on the API Keys tab for the Gemini key banner', () => {
    assert.deepEqual(GEMINI_KEY_REQUIRED_SETTINGS_DETAIL, {
      tab: 'apiKeys',
      reason: 'geminiKeyRequired',
    })
  })
})
