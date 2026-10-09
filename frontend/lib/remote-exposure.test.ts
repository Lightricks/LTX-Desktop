import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  isRemoteExposureEnabled,
  nextRemoteExposure,
} from './remote-exposure.ts'

describe('remote exposure', () => {
  it('treats lan as enabled and off as disabled', () => {
    assert.equal(isRemoteExposureEnabled('lan'), true)
    assert.equal(isRemoteExposureEnabled('off'), false)
  })

  it('toggles between off and lan', () => {
    assert.equal(nextRemoteExposure('off'), 'lan')
    assert.equal(nextRemoteExposure('lan'), 'off')
    assert.equal(nextRemoteExposure(nextRemoteExposure('off')), 'off')
  })
})
