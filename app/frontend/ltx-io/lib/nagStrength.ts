/**
 * Per-run strength of the negative prompt (NAG), behind the "Per-run strength controls" setting.
 *
 * NAG has three knobs (backend `runtime_config/nag_limits.py`):
 * - scale extrapolates away from the negative: Z⁺ + (scale − 1)(Z⁺ − Z⁻).
 * - τ caps that per token (‖Ẑ‖₁ / ‖Z⁺‖₁ ≤ τ), so past a point more scale is clipped away.
 * - α blends the capped result back toward the positive: α·Ẑ + (1 − α)·Z⁺.
 * Because the cap sits between them, α moves the result more than scale does. In our A/B
 * runs scale 11 → 20 left clouds and a moon in place, and α 0.25 → 0.5 removed both.
 *
 * Too much breaks the scene instead: when the negative fights the prompt (a moon in a scene
 * "lit from above"), α 0.8, α 0.65 with τ 4, and scale 20 with α 0.5 and τ 5 all replaced
 * the scene with an unrelated one, and brought negated content back. α 0.65 alone was the
 * strongest setting that held on every prompt we ran; scenes the default already handled
 * stayed intact up to the strongest setting.
 *
 * "default" sends nothing, so the run keeps the Settings scale and the backend α and τ.
 */

export type NagPreset = "default" | "strong" | "max" | "custom";

export type NagKnobs = { scale: number; alpha: number; tau: number };

export type NagStrength = { preset: NagPreset; custom: NagKnobs };

/** The request fields every queued feature with a negative prompt accepts (`NagParams`). */
export type NagRequestParams = { nagScale?: number; nagAlpha?: number; nagTau?: number };

/** Backend defaults (`DEFAULT_SCALE`, `DEFAULT_ALPHA`, `DEFAULT_TAU`). Custom starts here. */
export const NAG_DEFAULT_KNOBS: NagKnobs = { scale: 11, alpha: 0.25, tau: 2.5 };

/**
 * Slider ranges. The backend accepts scale ≤ 100, α ≤ 1, τ ≤ 10. Scale stops at 30 here:
 * above about 11 it barely moves the result, and the NAG paper sees quality drop past it.
 * Scale starts at 1 because 1 means no guidance and 0 turns NAG off.
 */
export const NAG_KNOB_RANGES = {
  scale: { min: 1, max: 30, step: 0.5 },
  // Above about 0.65 a scene can collapse (see above), so the slider stops at 0.7.
  alpha: { min: 0, max: 0.7, step: 0.05 },
  tau: { min: 1, max: 10, step: 0.5 },
} as const satisfies Record<keyof NagKnobs, { min: number; max: number; step: number }>;

/**
 * Preset overrides, from a sweep over four prompts at 540p (one seed each), plus a second seed
 * where Strong again cleared clouds the default left. Best effort: Max is not yet multi-seed
 * checked, and 720p/1080p may move where a scene collapses. Revisit with more seeds there.
 */
const PRESET_PARAMS: Record<Exclude<NagPreset, "custom">, NagRequestParams> = {
  default: {},
  strong: { nagAlpha: 0.5 },
  max: { nagAlpha: 0.65 },
};

export const NAG_PRESETS: { id: NagPreset; label: string; description: string }[] = [
  { id: "default", label: "Default", description: "The strength from Settings." },
  {
    id: "strong",
    label: "Strong",
    description: "α 0.5. Removes things the default leaves in, such as clouds or a moon.",
  },
  {
    id: "max",
    label: "Max",
    description:
      "α 0.65. For what Strong leaves in, such as smeared motion. Can change the setting or framing.",
  },
  { id: "custom", label: "Custom", description: "Set the scale, α and τ yourself." },
];

export const DEFAULT_NAG_STRENGTH: NagStrength = {
  preset: "default",
  custom: NAG_DEFAULT_KNOBS,
};

function clamp(value: number, range: { min: number; max: number }): number {
  if (!Number.isFinite(value)) return range.min;
  return Math.min(range.max, Math.max(range.min, value));
}

/** The NAG fields to send for this strength. Empty for the default preset. */
export function nagStrengthParams(strength: NagStrength): NagRequestParams {
  if (strength.preset !== "custom") return PRESET_PARAMS[strength.preset];
  return {
    nagScale: clamp(strength.custom.scale, NAG_KNOB_RANGES.scale),
    nagAlpha: clamp(strength.custom.alpha, NAG_KNOB_RANGES.alpha),
    nagTau: clamp(strength.custom.tau, NAG_KNOB_RANGES.tau),
  };
}

/**
 * Add the strength to a queued-generation create body. Leaves the body alone when the run
 * has no negative prompt text (NAG stays off, so the knobs would do nothing) or when the
 * preset is the default.
 */
export function injectNagStrength<T extends { params: Record<string, unknown> }>(
  body: T,
  strength: NagStrength,
): T {
  const negativePrompt = body.params.negativePrompt;
  if (typeof negativePrompt !== "string" || negativePrompt.trim() === "") return body;
  const overrides = nagStrengthParams(strength);
  if (Object.keys(overrides).length === 0) return body;
  return { ...body, params: { ...body.params, ...overrides } };
}

/** Short summary for the collapsed row, e.g. "Strong" or "Custom · scale 11, α 0.4, τ 2.5". */
export function nagStrengthSummary(strength: NagStrength): string {
  const preset = NAG_PRESETS.find((entry) => entry.id === strength.preset);
  if (strength.preset !== "custom") return preset?.label ?? "Default";
  const { nagScale, nagAlpha, nagTau } = nagStrengthParams(strength);
  return `Custom · scale ${nagScale}, α ${nagAlpha}, τ ${nagTau}`;
}
