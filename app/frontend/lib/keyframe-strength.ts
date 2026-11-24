/** Newly placed GenSpace stills. Loose enough for Distilled all-guide interpolation. */
export const DEFAULT_KEYFRAME_STRENGTH = 0.7

/** Omitted/invalid persist and HTTP. Matches the backend KeyframeInput Field default. */
export const MISSING_KEYFRAME_STRENGTH = 1

/** ArrowUp / ArrowDown and the strength rail step in 5% increments. */
export const KEYFRAME_STRENGTH_STEP = 0.05

/** Persist, restore, and HTTP all go through this so a bad float cannot 422 or fail project parse. */
export function clampKeyframeStrength(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return MISSING_KEYFRAME_STRENGTH
  return Math.min(1, Math.max(0, value))
}

function clampUnit(value: number): number {
  return Math.min(1, Math.max(0, value))
}

function roundStrength(value: number): number {
  return Math.round(clampUnit(value) * 100) / 100
}

/** Top of the rail is 1 (full lock), bottom is 0 (no lock). */
export function strengthFromPointer(
  clientY: number,
  railRect: { top: number; height: number },
): number {
  if (!(railRect.height > 0) || !Number.isFinite(clientY)) return DEFAULT_KEYFRAME_STRENGTH
  return roundStrength(1 - (clientY - railRect.top) / railRect.height)
}

export function nudgeKeyframeStrength(strength: number, direction: -1 | 1): number {
  const current = typeof strength === 'number' && Number.isFinite(strength)
    ? clampUnit(strength)
    : DEFAULT_KEYFRAME_STRENGTH
  return roundStrength(current + direction * KEYFRAME_STRENGTH_STEP)
}

export function formatKeyframeStrength(strength: number): string {
  const current = typeof strength === 'number' && Number.isFinite(strength)
    ? clampUnit(strength)
    : DEFAULT_KEYFRAME_STRENGTH
  return `${Math.round(current * 100)}%`
}
