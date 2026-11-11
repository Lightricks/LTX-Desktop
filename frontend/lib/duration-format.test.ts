import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  FINISHING_LABEL,
  formatEstimatedDuration,
  formatRemainingTime,
  formatShortRemainingTime,
} from './duration-format.ts'

const UNUSABLE = [null, undefined, -1, Number.NaN, Number.POSITIVE_INFINITY]

describe('remaining time labels', () => {
  it('returns null for a missing or unusable value', () => {
    for (const value of UNUSABLE) {
      assert.equal(formatRemainingTime(value), null)
      assert.equal(formatShortRemainingTime(value), null)
    }
  })

  it('says finishing at zero', () => {
    assert.equal(formatRemainingTime(0), FINISHING_LABEL)
    assert.equal(formatShortRemainingTime(0), FINISHING_LABEL)
  })

  it('rounds seconds up to five, with a floor of five', () => {
    assert.equal(formatRemainingTime(1), 'About 5 sec left')
    assert.equal(formatRemainingTime(6), 'About 10 sec left')
    assert.equal(formatShortRemainingTime(59), '~60s left')
  })

  it('uses minutes, then hours', () => {
    assert.equal(formatRemainingTime(60), 'About 1 min left')
    assert.equal(formatRemainingTime(125), 'About 2 min left')
    assert.equal(formatRemainingTime(3600), 'About 1 hr left')
    assert.equal(formatRemainingTime(3600 + 25 * 60), 'About 1 hr 25 min left')
    assert.equal(formatShortRemainingTime(125), '~2m left')
    assert.equal(formatShortRemainingTime(3600), '~1h left')
    assert.equal(formatShortRemainingTime(3600 + 25 * 60), '~1h 25m left')
  })
})

describe('formatEstimatedDuration', () => {
  it('returns null for zero or an unusable value', () => {
    for (const value of [...UNUSABLE, 0]) {
      assert.equal(formatEstimatedDuration(value), null)
    }
  })

  it('rounds seconds to five, with a floor of ten, and has no "left"', () => {
    assert.equal(formatEstimatedDuration(3), 'About 10 sec')
    assert.equal(formatEstimatedDuration(33), 'About 35 sec')
  })

  it('uses minutes, then hours', () => {
    assert.equal(formatEstimatedDuration(90), 'About 2 min')
    assert.equal(formatEstimatedDuration(7200), 'About 2 hr')
    assert.equal(formatEstimatedDuration(7200 + 10 * 60), 'About 2 hr 10 min')
  })
})
