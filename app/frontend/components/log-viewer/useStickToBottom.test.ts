import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { getScrollState } from './useStickToBottom.ts'

describe('getScrollState', () => {
  it('sticks and hides the button at the bottom', () => {
    assert.deepEqual(getScrollState(0), { sticks: true, showJumpButton: false })
  })

  it('still sticks within the snap distance', () => {
    assert.equal(getScrollState(16).sticks, true)
  })

  it('releases the stick once the user scrolls past the snap distance', () => {
    assert.deepEqual(getScrollState(17), { sticks: false, showJumpButton: false })
  })

  it('shows the jump button only far from the bottom', () => {
    assert.equal(getScrollState(160).showJumpButton, false)
    assert.deepEqual(getScrollState(161), { sticks: false, showJumpButton: true })
  })
})
