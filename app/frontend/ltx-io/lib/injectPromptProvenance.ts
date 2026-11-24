import type { PromptProvenance } from "@/lib/prompt-provenance";

/** Attach prompt provenance to queued-generation create bodies. */
export function injectPromptProvenance<
  T extends { params: Record<string, unknown> },
>(body: T, promptProvenance: PromptProvenance): T {
  return {
    ...body,
    params: {
      ...body.params,
      promptProvenance,
    },
  };
}
