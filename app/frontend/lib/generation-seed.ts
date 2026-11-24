/** Inclusive max; matches the backend `MAX_GENERATION_SEED` (32-bit signed). */
export const MAX_GENERATION_SEED = 2_147_483_647;

export function randomGenerationSeed(excluding?: number): number {
  const span = MAX_GENERATION_SEED + 1;
  let next = Math.floor(Math.random() * span);
  if (excluding != null && next === excluding) {
    next = (next + 1) % span;
  }
  return next;
}

export function clampGenerationSeed(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(MAX_GENERATION_SEED, Math.max(0, Math.trunc(value)));
}

/** Whole integer only (optional minus). Empty or mixed text returns null. */
export function parseGenerationSeedInput(text: string): number | null {
  const trimmed = text.trim();
  if (trimmed === "" || !/^-?\d+$/.test(trimmed)) return null;
  const negative = trimmed.startsWith("-");
  const digits = (negative ? trimmed.slice(1) : trimmed).replace(/^0+/, "") || "0";
  // Number() overflows past ~1e308 to Infinity, and clamp would then return 0.
  // Anything wider than the 10-digit max is already out of range.
  if (digits.length > 10) return negative ? 0 : MAX_GENERATION_SEED;
  return clampGenerationSeed(Number(negative ? `-${digits}` : digits));
}

/** Roll after a successful create only when that submit was unlocked and the field is unchanged. */
export function shouldRollSeedAfterSuccess(
  submitted: { seed: number; locked: boolean },
  currentSeed: number,
): boolean {
  return !submitted.locked && currentSeed === submitted.seed;
}
