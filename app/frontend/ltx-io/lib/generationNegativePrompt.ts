import { readGenerationParams } from "./resultsFeedModel.ts";

/**
 * The "Negative prompt: …" line for a run, or null when its row has no negative prompt
 * field (the feature has none).
 *
 * The model only sees a negative prompt through NAG. The backend decides at enqueue time
 * whether NAG runs and writes it into the row: `nagScale` stays set above 0 when NAG is on,
 * and is dropped (or 0) when NAG is off (empty text, strength 0, API text encoding, Torch
 * Compile). The default negative prompt is never sent. So the text counts as sent only when
 * `nagScale` is above 0. The backend sends the trimmed text. The scale shows only then too.
 * When NAG is on the row also holds the resolved `nagAlpha` and `nagTau`; they show after the
 * scale, so runs made with different strengths can be told apart.
 */
export function generationNegativePromptLine(spec: unknown): string | null {
  const params = readGenerationParams(spec);
  const negativePrompt = params?.negativePrompt;
  if (typeof negativePrompt !== "string") return null;
  const text = negativePrompt.trim();
  const nagScale = params?.nagScale;
  if (text && typeof nagScale === "number" && nagScale > 0) {
    const strength = [`scale: ${nagScale}`];
    if (typeof params?.nagAlpha === "number") strength.push(`α: ${params.nagAlpha}`);
    if (typeof params?.nagTau === "number") strength.push(`τ: ${params.nagTau}`);
    return `Negative prompt (${strength.join(", ")}): ${text}`;
  }
  return text ? "Negative prompt: (none, NAG off)" : "Negative prompt: (none)";
}
