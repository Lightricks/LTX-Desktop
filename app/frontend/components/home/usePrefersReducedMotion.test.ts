import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  matchesReducedMotion,
  REDUCED_MOTION_QUERY,
  type ReducedMotionWindow,
} from './usePrefersReducedMotion.ts'

function windowWith(matches: boolean, queries: string[]): ReducedMotionWindow {
  return {
    matchMedia(query: string) {
      queries.push(query)
      return {
        matches,
        addEventListener: () => {},
        removeEventListener: () => {},
      }
    },
  }
}

describe('matchesReducedMotion', () => {
  it('reads the reduced-motion preference from matchMedia', () => {
    const queries: string[] = []
    assert.equal(matchesReducedMotion(windowWith(true, queries)), true)
    assert.deepEqual(queries, [REDUCED_MOTION_QUERY])
  })

  it('reports motion as allowed when the preference is not set', () => {
    assert.equal(matchesReducedMotion(windowWith(false, [])), false)
  })

  it('falls back to motion allowed without a window or matchMedia', () => {
    assert.equal(matchesReducedMotion(undefined), false)
    assert.equal(matchesReducedMotion({}), false)
  })
})
