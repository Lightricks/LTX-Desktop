import { isHomeFeatureId, type HomeFeatureId } from "../../../../lib/home-features.ts";

/**
 * Deepest match carrying a feature handle. `UIMatch.handle` is `unknown`.
 */
export function selectHomeFeatureId(
  matches: readonly { readonly handle?: unknown }[],
): HomeFeatureId | null {
  for (let i = matches.length - 1; i >= 0; i -= 1) {
    const handle = matches[i]?.handle;
    if (handle == null || typeof handle !== "object") continue;
    const value = (handle as { homeFeatureId?: unknown }).homeFeatureId;
    if (typeof value === "string" && isHomeFeatureId(value)) return value;
  }
  return null;
}
