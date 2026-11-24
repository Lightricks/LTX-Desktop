// SRT subtitle format parsing and export utilities

export interface SrtCue {
  index: number
  startTime: number  // in seconds
  endTime: number    // in seconds
  text: string
  color?: string     // extracted from <font color=...> tags if present
}

/**
 * Parse SRT timestamp to seconds
 * Format: HH:MM:SS,mmm (e.g. "00:01:23,456")
 */
function parseTimestamp(ts: string): number {
  const match = ts.trim().match(/(\d{2}):(\d{2}):(\d{2})[,.](\d{3})/)
  if (!match) return 0
  const [, h, m, s, ms] = match
  return parseInt(h) * 3600 + parseInt(m) * 60 + parseInt(s) + parseInt(ms) / 1000
}

/**
 * Convert seconds to SRT timestamp
 * Returns format: HH:MM:SS,mmm
 */
function formatTimestamp(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  const ms = Math.round((seconds % 1) * 1000)
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')},${String(ms).padStart(3, '0')}`
}

/**
 * Strip HTML/rich-text tags from SRT text.
 * Handles <font color=...>, <b>, <i>, <u>, and any other HTML tags.
 * Extracts the first color value found (if any).
 */
function stripTags(text: string): { clean: string; color?: string } {
  let color: string | undefined

  // Extract color from <font color=...> (Premiere format: color=#RRGGBBAA or #RRGGBB)
  const colorMatch = text.match(/<font\s+color\s*=\s*["']?([^"'>]+)["']?\s*>/i)
  if (colorMatch) {
    let c = colorMatch[1].trim()
    // Premiere sometimes outputs 8-char hex (#RRGGBBAA) — convert to standard 6-char
    if (/^#[0-9A-Fa-f]{8}$/.test(c)) {
      c = c.slice(0, 7) // drop the alpha suffix
    }
    color = c
  }

  // Strip all HTML tags
  const clean = text
    .replace(/<[^>]+>/g, '')   // remove tags
    .replace(/\n\s*\n/g, '\n') // collapse blank lines left by removed tags
    .trim()

  return { clean, color }
}

// Threshold: cues shorter than this (in seconds) are considered "pre-cues" / fade markers
const PRE_CUE_THRESHOLD = 0.1 // 100ms

/**
 * Parse an SRT file content string into an array of cues.
 *
 * Handles:
 * - Standard SRT format
 * - Premiere Pro SRT with <font color=...> tags
 * - Premiere "pre-cue" pairs (near-zero-duration fade-in marker + real cue)
 *   → merged into a single cue using the pre-cue's start time and the real cue's end time
 */
export function parseSrt(content: string): SrtCue[] {
  const rawCues: SrtCue[] = []
  
  // Normalize line endings
  const normalized = content.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim()
  
  // Split into blocks separated by empty lines
  const blocks = normalized.split(/\n\n+/)
  
  for (const block of blocks) {
    const lines = block.trim().split('\n')
    if (lines.length < 3) continue
    
    // First line: index number
    const index = parseInt(lines[0].trim())
    if (isNaN(index)) continue
    
    // Second line: timestamps (start --> end)
    const timeParts = lines[1].split('-->')
    if (timeParts.length !== 2) continue
    
    const startTime = parseTimestamp(timeParts[0])
    const endTime = parseTimestamp(timeParts[1])
    
    if (endTime <= startTime) continue
    
    // Remaining lines: subtitle text (strip HTML tags)
    const rawText = lines.slice(2).join('\n').trim()
    if (!rawText) continue
    
    const { clean, color } = stripTags(rawText)
    if (!clean) continue
    
    rawCues.push({ index, startTime, endTime, text: clean, color })
  }
  
  // --- Merge Premiere-style pre-cue pairs ---
  // Pattern: a near-zero-duration cue immediately followed by a cue with the same text.
  // The first cue's start time is the real start; the second cue's end time is the real end.
  const merged: SrtCue[] = []
  let i = 0
  while (i < rawCues.length) {
    const cur = rawCues[i]
    const next = rawCues[i + 1]
    
    const curDuration = cur.endTime - cur.startTime
    
    if (
      next &&
      curDuration <= PRE_CUE_THRESHOLD &&
      cur.text === next.text &&
      Math.abs(cur.endTime - next.startTime) < 0.1 // consecutive
    ) {
      // Merge: use pre-cue's start + real cue's end
      merged.push({
        index: cur.index,
        startTime: cur.startTime,
        endTime: next.endTime,
        text: cur.text,
        color: cur.color || next.color,
      })
      i += 2 // skip both
    } else if (curDuration <= PRE_CUE_THRESHOLD) {
      // Standalone near-zero cue with no matching follow-up — skip it (likely orphan pre-cue)
      i++
    } else {
      merged.push(cur)
      i++
    }
  }
  
  // Re-index
  return merged.map((cue, idx) => ({ ...cue, index: idx + 1 }))
}

/**
 * Export an array of cues to SRT format string
 */
export function exportSrt(cues: { startTime: number; endTime: number; text: string }[]): string {
  // Sort by start time
  const sorted = [...cues].sort((a, b) => a.startTime - b.startTime)
  
  return sorted.map((cue, i) => {
    return `${i + 1}\n${formatTimestamp(cue.startTime)} --> ${formatTimestamp(cue.endTime)}\n${cue.text}`
  }).join('\n\n') + '\n'
}
