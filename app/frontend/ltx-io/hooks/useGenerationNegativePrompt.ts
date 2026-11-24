import { generationNegativePromptLine } from "../lib/generationNegativePrompt.ts";
import type { Generation } from "../lib/resultsFeedModel";
import { useFeatureFlagEnabled } from "./useFeatureFlagEnabled.ts";

/**
 * The "Negative prompt: …" line for a generation, behind the `showGenerationNegativePrompt`
 * flag. Null when the flag is off or the row has no negative prompt field.
 */
export function useGenerationNegativePrompt(generation: Generation): string | null {
  const enabled = useFeatureFlagEnabled("showGenerationNegativePrompt");
  return enabled ? generationNegativePromptLine(generation.spec) : null;
}
