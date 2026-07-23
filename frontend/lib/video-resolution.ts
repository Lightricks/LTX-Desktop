// Resolution tier options for local retake/extend. Offers the source resolution plus
// standard lower tiers (named by short edge: 1080p / 720p / 540p); the backend snaps the
// chosen size to a valid (÷32, not-upscaled) resolution. Local only — the cloud preserves
// source resolution.

export interface ResolutionOption {
  key: string
  label: string
  // null = "Original" (backend uses the source resolution, still ÷32-corrected).
  width: number | null
  height: number | null
}

const STANDARD_TIERS = [1080, 720, 540]
// A source whose short edge is within this of a tier is labelled as that tier (so a
// 1088px source still reads "1080p"), and that tier is dropped from the lower list.
const MATCH_TOLERANCE = 0.03

export function resolutionOptions(width: number, height: number): ResolutionOption[] {
  if (!width || !height) return []
  const shortEdge = Math.min(width, height)
  const longEdge = Math.max(width, height)
  const portrait = height > width

  // Name "Original" by the standard tier it matches (within tolerance), else its own height.
  const matchedTier = STANDARD_TIERS.find((t) => Math.abs(t - shortEdge) <= shortEdge * MATCH_TOLERANCE)
  const originalTier = matchedTier ?? shortEdge

  const options: ResolutionOption[] = [
    { key: 'original', label: `${originalTier}p (Original)`, width: null, height: null },
  ]
  for (const tier of STANDARD_TIERS) {
    // Only smaller tiers, and drop the one that already maps to Original.
    if (tier >= shortEdge || tier === originalTier) continue
    const long = Math.round((longEdge * tier) / shortEdge)
    options.push({
      key: String(tier),
      label: `${tier}p`,
      width: portrait ? tier : long,
      height: portrait ? long : tier,
    })
  }
  return options
}
