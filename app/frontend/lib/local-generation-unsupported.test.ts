import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  LOCAL_GENERATION_UNSUPPORTED_BODY,
  LOCAL_GENERATION_UNSUPPORTED_TITLE,
} from './local-generation-unsupported.ts'

describe('unsupported copy', () => {
  it('does not send the user to an API key', () => {
    const text = `${LOCAL_GENERATION_UNSUPPORTED_TITLE} ${LOCAL_GENERATION_UNSUPPORTED_BODY}`.toLowerCase()
    assert.equal(text.includes('api'), false)
    assert.equal(text.includes('key'), false)
  })
})
