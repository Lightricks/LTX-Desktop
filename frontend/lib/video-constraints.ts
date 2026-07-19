// Client-side guard mirroring the LTX API's input constraints, so we reject bad source
// videos instantly instead of after a slow upload + API round-trip. Dimensions come from
// the <video> element (videoWidth/videoHeight); the backend/API remain the authority.
//
// No aspect-ratio check: the API accepts near-16:9 sizes the docs don't list (e.g.
// 1920x1024) and ltx-studio doesn't pre-check aspect either, so a strict check
// false-rejects working videos. The API stays the authority for aspect.

export interface VideoDimensions {
  width: number
  height: number
  duration: number
}

// API: "Maximum resolution 3840x2160 (4K)", "Minimum frame count 73 (~3s at 24fps)".
const MAX_LONG_EDGE = 3840
const MAX_SHORT_EDGE = 2160
const MIN_DURATION_S = 3

export function validateVideoSource({ width, height, duration }: VideoDimensions): string | null {
  // Dimensions not known yet (metadata still loading) — don't block prematurely.
  if (!width || !height) return null
  const longEdge = Math.max(width, height)
  const shortEdge = Math.min(width, height)
  if (longEdge > MAX_LONG_EDGE || shortEdge > MAX_SHORT_EDGE) {
    return `Video is larger than 4K (${width}×${height}). Maximum supported is 3840×2160.`
  }
  if (duration && duration < MIN_DURATION_S) {
    return `Video is too short (${duration.toFixed(1)}s). It must be at least ${MIN_DURATION_S} seconds.`
  }
  return null
}
