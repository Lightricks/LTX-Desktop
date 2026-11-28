import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { describe, it } from 'node:test'

import {
  IC_LORA_SOURCE_TOO_LARGE,
  LOCAL_GENERATION_RAM_SHORTFALL,
  LOCAL_GENERATION_UNSUPPORTED,
  LTX_API_KEY_MISSING,
  LTX_INVALID_API_KEY,
  VIDEO_JOB_TOO_LARGE,
  createLocalGenerationError,
  generationErrorFromApiFailure,
  isActionableLocalCode,
} from './generation-errors.ts'

describe('createLocalGenerationError', () => {
  it('preserves every actionable local code the dialog passes through verbatim', () => {
    for (const code of [
      IC_LORA_SOURCE_TOO_LARGE,
      VIDEO_JOB_TOO_LARGE,
      LOCAL_GENERATION_UNSUPPORTED,
      LOCAL_GENERATION_RAM_SHORTFALL,
    ]) {
      assert.equal(createLocalGenerationError('backend sentence', code).error.code, code)
    }
  })

  it('folds unknown codes to LOCAL_GENERATION_ERROR', () => {
    assert.equal(
      createLocalGenerationError('backend sentence', 'BOGUS_CODE').error.code,
      'LOCAL_GENERATION_ERROR',
    )
  })
})

describe('budget-miss contract', () => {
  it('matches shared/budget-miss-codes.json exactly, both directions', () => {
    const contract = JSON.parse(
      readFileSync(new URL('../../shared/budget-miss-codes.json', import.meta.url), 'utf8'),
    ) as string[]
    const localConsts = [
      IC_LORA_SOURCE_TOO_LARGE,
      VIDEO_JOB_TOO_LARGE,
      LOCAL_GENERATION_UNSUPPORTED,
      LOCAL_GENERATION_RAM_SHORTFALL,
    ]
    assert.deepEqual([...localConsts].sort(), [...contract].sort())
    for (const code of contract) {
      assert.equal(isActionableLocalCode(code), true)
    }
    assert.equal(isActionableLocalCode('BOGUS_CODE'), false)
  })
})

describe('generationErrorFromApiFailure', () => {
  it('keeps a rejected LTX key typed so the dialog can open Settings', () => {
    const error = generationErrorFromApiFailure({
      code: LTX_INVALID_API_KEY,
      message: 'This LTX API key isn’t valid.',
    })
    assert.equal(error.status, '4XX')
    assert.equal(error.error.code, LTX_INVALID_API_KEY)
  })

  it('keeps a missing LTX key typed the same way', () => {
    const error = generationErrorFromApiFailure({
      code: LTX_API_KEY_MISSING,
      message: 'Add an LTX API key to generate.',
    })
    assert.equal(error.status, '4XX')
    assert.equal(error.error.code, LTX_API_KEY_MISSING)
  })

  it('leaves other failures as a local generation error', () => {
    const error = generationErrorFromApiFailure({
      code: 'HTTP_500',
      message: 'Retake completed but no local video file was returned',
    })
    assert.equal(error.status, 'default')
    assert.equal(error.error.code, 'LOCAL_GENERATION_ERROR')
    assert.equal(error.error.message, 'Retake completed but no local video file was returned')
  })
})
