export { formatBytes } from '../../shared/format-bytes.ts'

/** Settings base-model row title (API may still return bare version numbers until backend reload). */
export function formatLtxBaseModelLabel(label: string): string {
  const trimmed = label.trim()
  if (/^LTX\b/i.test(trimmed)) return trimmed
  return trimmed ? `LTX ${trimmed}` : 'LTX'
}

// Human-readable duration, e.g. 45 -> "45s", 90 -> "2m", 3700 -> "1h 2m".
// Unknown or non-positive input renders as "--" (used for download ETAs).
export function formatTimeRemaining(seconds: number): string {
  if (!seconds || !isFinite(seconds) || seconds <= 0) return '--'
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`
}
