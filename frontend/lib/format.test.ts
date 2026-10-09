import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { formatBytes, formatLtxBaseModelLabel, formatTimeRemaining } from './format.ts'

describe('formatBytes', () => {
  it('formats zero', () => {
    assert.equal(formatBytes(0), '0 B')
  })

  it('formats megabytes with one decimal', () => {
    assert.equal(formatBytes(996_000_000), '949.9 MB')
  })

  it('formats terabytes instead of overflowing the unit list', () => {
    assert.equal(formatBytes(1.8 * 1024 ** 4), '1.8 TB')
  })
})

describe('formatLtxBaseModelLabel', () => {
  it('prefixes a bare version number', () => {
    assert.equal(formatLtxBaseModelLabel('2.5'), 'LTX 2.5')
  })

  it('does not double the LTX prefix', () => {
    assert.equal(formatLtxBaseModelLabel('LTX 2.5'), 'LTX 2.5')
  })

  it('falls back when the API sends an empty label', () => {
    assert.equal(formatLtxBaseModelLabel('  '), 'LTX')
  })
})

describe('formatTimeRemaining', () => {
  it('renders unknown or non-positive durations as --', () => {
    assert.equal(formatTimeRemaining(0), '--')
    assert.equal(formatTimeRemaining(-5), '--')
    assert.equal(formatTimeRemaining(NaN), '--')
    assert.equal(formatTimeRemaining(Infinity), '--')
  })

  it('renders seconds, minutes, and hours', () => {
    assert.equal(formatTimeRemaining(45), '45s')
    assert.equal(formatTimeRemaining(90), '2m')
    assert.equal(formatTimeRemaining(3700), '1h 2m')
  })
})
