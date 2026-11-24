import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  DEFAULT_KEYFRAME_STRENGTH,
  formatKeyframeStrength,
  nudgeKeyframeStrength,
  strengthFromPointer,
} from './keyframe-strength.ts'

describe('strengthFromPointer', () => {
  const rail = { top: 10, height: 100 }

  it('maps the top of the rail to full strength', () => {
    assert.equal(strengthFromPointer(10, rail), 1)
  })

  it('maps the bottom of the rail to zero strength', () => {
    assert.equal(strengthFromPointer(110, rail), 0)
  })

  it('maps the midpoint to 0.5', () => {
    assert.equal(strengthFromPointer(60, rail), 0.5)
  })

  it('clamps above the rail to 1 and below it to 0', () => {
    assert.equal(strengthFromPointer(0, rail), 1)
    assert.equal(strengthFromPointer(200, rail), 0)
  })

  it('returns the new-still default when the rail has no height', () => {
    assert.equal(strengthFromPointer(10, { top: 10, height: 0 }), DEFAULT_KEYFRAME_STRENGTH)
  })
})

describe('nudgeKeyframeStrength', () => {
  it('steps by five percent', () => {
    assert.equal(nudgeKeyframeStrength(0.7, 1), 0.75)
    assert.equal(nudgeKeyframeStrength(0.7, -1), 0.65)
  })

  it('clamps at 0 and 1', () => {
    assert.equal(nudgeKeyframeStrength(0.02, -1), 0)
    assert.equal(nudgeKeyframeStrength(0.98, 1), 1)
  })
})

describe('formatKeyframeStrength', () => {
  it('renders owned strength as a percent', () => {
    assert.equal(formatKeyframeStrength(0.7), '70%')
    assert.equal(formatKeyframeStrength(0), '0%')
    assert.equal(formatKeyframeStrength(1), '100%')
  })
})
