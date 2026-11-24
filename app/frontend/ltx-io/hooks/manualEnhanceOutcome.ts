import { MISSING_GEMINI_KEY_MESSAGE } from "../../lib/enhance-gemini-key.ts";
import {
  type PromptHistoryEntry,
  appendEnhanceResult,
  resolvePromptProvenance,
} from "../../lib/prompt-provenance.ts";

/**
 * The message of a failed Enhance. A missing or invalid Gemini key also asks the app for the
 * key. Without that, the message names where to add it.
 */
export function enhanceFailureMessage(
  error: { code: string; message: string },
  onGeminiKeyRequired?: () => void,
): string {
  const keyError =
    error.code === "GEMINI_INVALID_API_KEY" || error.code === "GEMINI_API_KEY_MISSING";
  if (!keyError) return error.message;
  onGeminiKeyRequired?.();
  return onGeminiKeyRequired ? error.message : MISSING_GEMINI_KEY_MESSAGE;
}

/** The prompt history after an Enhance: the typed prompt is kept, the enhanced one is added. */
export function historyAfterEnhance(input: {
  history: PromptHistoryEntry[];
  index: number;
  sourcePrompt: string;
  sourceNegativePrompt?: string;
  enhancedPrompt: string;
}) {
  return appendEnhanceResult({
    ...input,
    sourceProvenance: resolvePromptProvenance(
      input.sourcePrompt,
      input.history,
      input.index,
      input.sourceNegativePrompt,
    ),
  });
}
