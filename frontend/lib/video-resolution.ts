// Resolution tier options for local retake/extend. Offers the source resolution plus
// standard lower tiers (named by short edge: 1080p / 720p / 540p). Grid sizes such as
// 576/704/1088 map to the nearest named tier. The backend snaps the chosen size to a
// valid (÷32, not-upscaled) resolution. Local only — the cloud preserves source resolution.

export interface ResolutionOption {
  key: string
  label: string
  // null = "Original" (backend uses the source resolution, still ÷32-corrected).
  width: number | null
  height: number | null
}

const STANDARD_TIERS = [1080, 720, 540]
const NAMED_TIERS = [2160, 1440, 1080, 720, 540] as const

/** Map a pixel short-edge (incl. /64 grid sizes like 576, 704, 1088) to the picker tier name. */
export function namedResolutionTier(shortEdge: number): (typeof NAMED_TIERS)[number] {
  return NAMED_TIERS.reduce((best, tier) =>
    Math.abs(tier - shortEdge) < Math.abs(best - shortEdge) ? tier : best,
  )
}

export function namedResolutionDisplayName(tier: number): string {
  return tier >= 2160 ? '4K' : `${tier}p`
}

/** Display label for generation resolution ids (`2160p` → `4K`). */
export function videoGenerationResolutionLabel(resolution: string): string {
  return resolution === '2160p' ? '4K' : resolution
}

export function resolutionOptions(width: number, height: number): ResolutionOption[] {
  if (!width || !height) return []
  const shortEdge = Math.min(width, height)
  const longEdge = Math.max(width, height)
  const portrait = height > width

  const originalTier = namedResolutionTier(shortEdge)

  const options: ResolutionOption[] = [
    { key: 'original', label: `${namedResolutionDisplayName(originalTier)} (Original)`, width: null, height: null },
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
