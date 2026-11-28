import { useCallback, useEffect, useMemo, useState } from 'react'
import type { LogEntry } from './logEntries'
import {
  compileSearch,
  findMatchRanges,
  stepMatchIndex,
  type CompiledSearch,
  type MatchRange,
} from './logSearch'

const SEARCH_DEBOUNCE_MS = 300

/** Search field state. The compiled search follows the input after a short debounce. */
export function useSearchInput() {
  const [input, setInput] = useState('')
  const [query, setQuery] = useState('')
  const [matchCase, setMatchCase] = useState(false)
  const [useRegex, setUseRegex] = useState(false)

  // Debounce the raw input so typing does not fire an IPC round-trip (a full
  // untruncated log read while searching) on every keystroke.
  useEffect(() => {
    const timer = setTimeout(() => setQuery(input.trim()), SEARCH_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [input])

  const search = useMemo(
    () => compileSearch(query, { matchCase, useRegex }),
    [query, matchCase, useRegex],
  )

  // Clearing skips the debounce, so the highlights and the match count go away at once.
  const clear = useCallback(() => {
    setInput('')
    setQuery('')
  }, [])

  return { input, setInput, clear, matchCase, setMatchCase, useRegex, setUseRegex, search }
}

const NO_RANGES: MatchRange[][] = []
const NO_LINES: number[] = []

/** Finds the matching entries and tracks which match is current. */
export function useSearchMatches(entries: LogEntry[], search: CompiledSearch) {
  const lineRanges = useMemo(
    () =>
      search.kind === 'ok'
        ? entries.map((entry) => findMatchRanges(entry.text, search.regex))
        : NO_RANGES,
    [entries, search],
  )
  const matchLines = useMemo(() => {
    if (search.kind !== 'ok') return NO_LINES
    const lines: number[] = []
    lineRanges.forEach((ranges, index) => {
      if (ranges.length > 0) lines.push(index)
    })
    return lines
  }, [lineRanges, search])

  const [currentMatch, setCurrentMatch] = useState(0)
  // Bumps on every next/previous so the view scrolls even when the match does not change.
  const [scrollNonce, setScrollNonce] = useState(0)

  // A new search starts at the first match. New data for the same search (Refresh) or a
  // level change keeps the position. `safeCurrent` below clamps it to the new count.
  useEffect(() => setCurrentMatch(0), [search])

  const matchCount = matchLines.length
  const safeCurrent = matchCount > 0 ? Math.min(currentMatch, matchCount - 1) : -1

  const next = useCallback(() => {
    if (matchCount === 0) return
    setCurrentMatch((current) => stepMatchIndex(current, matchCount, 1))
    setScrollNonce((n) => n + 1)
  }, [matchCount])

  const previous = useCallback(() => {
    if (matchCount === 0) return
    setCurrentMatch((current) => stepMatchIndex(current, matchCount, -1))
    setScrollNonce((n) => n + 1)
  }, [matchCount])

  return {
    lineRanges,
    matchCount,
    currentMatchNumber: safeCurrent + 1,
    currentLine: safeCurrent >= 0 ? matchLines[safeCurrent] : -1,
    scrollNonce,
    next,
    previous,
  }
}
