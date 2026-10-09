import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import {
  LTX_API_KEY_MISSING,
  LTX_INVALID_API_KEY,
  generationErrorFromApiFailure,
} from './generation-errors.ts'

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
