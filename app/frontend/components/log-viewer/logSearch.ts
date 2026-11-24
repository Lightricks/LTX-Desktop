export type MatchRange = readonly [start: number, end: number]

export interface SearchOptions {
  matchCase: boolean
  useRegex: boolean
}

export type CompiledSearch =
  | { kind: 'none' }
  | { kind: 'invalid' }
  | { kind: 'ok'; query: string; regex: RegExp }

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

export function compileSearch(query: string, options: SearchOptions): CompiledSearch {
  if (!query) return { kind: 'none' }
  try {
    const source = options.useRegex ? query : escapeRegExp(query)
    return { kind: 'ok', query, regex: new RegExp(source, options.matchCase ? 'g' : 'gi') }
  } catch {
    return { kind: 'invalid' }
  }
}

/** Moves `current` by `delta` among `count` matches and wraps around at both ends. */
export function stepMatchIndex(current: number, count: number, delta: 1 | -1): number {
  return (((Math.min(current, count - 1) + delta) % count) + count) % count
}

/** Non-empty matches of `regex` in `line`. `regex` must have the global flag. */
export function findMatchRanges(line: string, regex: RegExp): MatchRange[] {
  const ranges: MatchRange[] = []
  regex.lastIndex = 0
  let match = regex.exec(line)
  while (match !== null) {
    if (match[0].length === 0) {
      // A pattern such as `a*` can match nothing. Step over it to avoid a loop.
      regex.lastIndex += 1
    } else {
      ranges.push([match.index, match.index + match[0].length])
    }
    match = regex.exec(line)
  }
  return ranges
}
