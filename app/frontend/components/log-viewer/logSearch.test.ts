import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { compileSearch, findMatchRanges, stepMatchIndex } from './logSearch.ts'

const plain = { matchCase: false, useRegex: false }

function rangesOf(line: string, query: string, options = plain) {
  const search = compileSearch(query, options)
  assert.equal(search.kind, 'ok')
  return search.kind === 'ok' ? findMatchRanges(line, search.regex) : []
}

describe('compileSearch', () => {
  it('returns none for an empty query', () => {
    assert.deepEqual(compileSearch('', plain), { kind: 'none' })
  })

  it('returns invalid for a broken regex', () => {
    assert.deepEqual(compileSearch('(', { ...plain, useRegex: true }), { kind: 'invalid' })
  })

  it('treats regex characters as text when regex mode is off', () => {
    assert.deepEqual(rangesOf('a.b axb', 'a.b'), [[0, 3]])
    assert.equal(compileSearch('(', plain).kind, 'ok')
  })

  it('uses regex syntax when regex mode is on', () => {
    assert.deepEqual(rangesOf('a.b axb', 'a.b', { ...plain, useRegex: true }), [
      [0, 3],
      [4, 7],
    ])
  })
})

describe('match case', () => {
  it('ignores case by default', () => {
    assert.equal(rangesOf('Error error ERROR', 'error').length, 3)
  })

  it('matches exact case when enabled', () => {
    assert.deepEqual(rangesOf('Error error ERROR', 'error', { ...plain, matchCase: true }), [[6, 11]])
  })
})

describe('findMatchRanges', () => {
  it('does not hang on a pattern that can match nothing', () => {
    const ranges = rangesOf('baaac', 'a*', { ...plain, useRegex: true })
    assert.deepEqual(ranges, [[1, 4]])
  })

  it('returns no ranges for a pattern that only matches nothing', () => {
    assert.deepEqual(rangesOf('abc', 'x*', { ...plain, useRegex: true }), [])
  })

  it('gives the same result when called twice with one regex', () => {
    const search = compileSearch('a', plain)
    assert.equal(search.kind, 'ok')
    if (search.kind !== 'ok') return
    assert.deepEqual(findMatchRanges('aa', search.regex), findMatchRanges('aa', search.regex))
  })
})

describe('stepMatchIndex', () => {
  it('moves forward and wraps to the first match', () => {
    assert.equal(stepMatchIndex(0, 3, 1), 1)
    assert.equal(stepMatchIndex(2, 3, 1), 0)
  })

  it('moves back and wraps to the last match', () => {
    assert.equal(stepMatchIndex(1, 3, -1), 0)
    assert.equal(stepMatchIndex(0, 3, -1), 2)
  })

  it('stays on the only match', () => {
    assert.equal(stepMatchIndex(0, 1, 1), 0)
    assert.equal(stepMatchIndex(0, 1, -1), 0)
  })

  it('clamps a stale index after the match list shrinks', () => {
    assert.equal(stepMatchIndex(9, 3, 1), 0)
    assert.equal(stepMatchIndex(9, 3, -1), 1)
  })
})
