import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { shouldShowLastFrameChip } from './genspace-last-frame.ts'

describe('shouldShowLastFrameChip', () => {
  it('shows the chip only for video with a first frame and a concrete duration', () => {
    assert.equal(
      shouldShowLastFrameChip({ mode: 'video', hasFirstFrame: false, duration: 5 }),
      false,
    )
    assert.equal(
      shouldShowLastFrameChip({ mode: 'video', hasFirstFrame: true, duration: null }),
      false,
    )
    assert.equal(
      shouldShowLastFrameChip({ mode: 'image', hasFirstFrame: true, duration: 5 }),
      false,
    )
    assert.equal(
      shouldShowLastFrameChip({ mode: 'video', hasFirstFrame: true, duration: 5 }),
      true,
    )
  })
})
