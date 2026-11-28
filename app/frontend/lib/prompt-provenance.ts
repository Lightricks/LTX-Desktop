// Where the prompt in the box came from, so Generate knows whether the backend may rewrite it.
// Generation enhances every typed prompt automatically; a prompt the user produced with the
// Enhance button and kept must be submitted exactly as it stands, or Generate would rewrite a
// rewrite. Editing the text makes it typed again, and so eligible again.
export type PromptProvenance = "typed" | "enhanced";

export interface PromptHistoryEntry {
  text: string;
  negativeText?: string;
  provenance: PromptProvenance;
}

export interface PromptHistoryState {
  history: PromptHistoryEntry[];
  index: number;
}

function sameNegativeText(entryNegative: string | undefined, liveNegative: string | undefined): boolean {
  if (liveNegative === undefined) return true;
  return (entryNegative ?? "") === liveNegative;
}

/** Derived, never stored: the live prompt text is the only thing an edit changes. */
export function resolvePromptProvenance(
  prompt: string,
  history: PromptHistoryEntry[],
  index: number,
  _negativePrompt?: string,
): PromptProvenance {
  const current = index >= 0 ? history[index] : undefined;
  if (current === undefined) return "typed";
  // Undo/redo puts an entry's exact text back, so matching text means we are still on that
  // entry. Any edit (including trailing whitespace) diverges from it and is the user's own text.
  // Editing only the negative prompt keeps the approved positive rewrite intact on Generate.
  return current.text === prompt ? current.provenance : "typed";
}

/**
 * Push [source (unless it is already the current entry), enhanced] onto the stack, dropping any
 * redo tail first — the same rule any undo/redo stack uses: a new action after an undo discards
 * the redone-away future.
 */
export function appendEnhanceResult(input: {
  history: PromptHistoryEntry[];
  index: number;
  sourcePrompt: string;
  sourceNegativePrompt?: string;
  sourceProvenance: PromptProvenance;
  enhancedPrompt: string;
}): PromptHistoryState {
  const truncated = input.history.slice(0, input.index + 1);
  const last = truncated[truncated.length - 1];
  const withSource =
    last !== undefined &&
    last.text === input.sourcePrompt &&
    sameNegativeText(last.negativeText, input.sourceNegativePrompt)
      ? truncated
      : [
          ...truncated,
          {
            text: input.sourcePrompt,
            ...(input.sourceNegativePrompt !== undefined
              ? { negativeText: input.sourceNegativePrompt }
              : {}),
            provenance: input.sourceProvenance,
          },
        ];
  const history = [
    ...withSource,
    {
      text: input.enhancedPrompt,
      ...(input.sourceNegativePrompt !== undefined
        ? { negativeText: input.sourceNegativePrompt }
        : {}),
      provenance: "enhanced" as const,
    },
  ];
  return { history, index: history.length - 1 };
}

/**
 * Seed the stack from a recovery marker so a reload mid-generation does not lose the fact that
 * the in-flight prompt was already enhanced — otherwise the next Generate enhances it again.
 * Markers written before provenance existed carry undefined and are treated as typed.
 */
export function historyForRestoredPrompt(
  prompt: string,
  provenance: PromptProvenance | undefined,
  negativePrompt?: string,
): PromptHistoryState {
  if (provenance !== "enhanced") return { history: [], index: -1 };
  return {
    history: [
      {
        text: prompt,
        ...(negativePrompt !== undefined ? { negativeText: negativePrompt } : {}),
        provenance: "enhanced",
      },
    ],
    index: 0,
  };
}

export function parsePromptProvenance(value: unknown): PromptProvenance | undefined {
  return value === "typed" || value === "enhanced" ? value : undefined;
}

/** Drop a rewrite if the box no longer holds the prompt that was sent. */
export function shouldApplyEnhanceResult(
  livePrompt: string,
  sourcePrompt: string,
  liveNegativePrompt?: string,
  sourceNegativePrompt?: string,
): boolean {
  return (
    livePrompt === sourcePrompt &&
    sameNegativeText(sourceNegativePrompt, liveNegativePrompt)
  );
}

/**
 * Keep Enhance undo/redo across form remounts. In-session history wins; otherwise seed from
 * the last generation when the box still holds that generation's prompt.
 */
export function restorePromptHistory(input: {
  stored: PromptHistoryState | undefined;
  prompt: string;
  negativePrompt?: string;
  lastGenerationPrompt: string | undefined;
  lastGenerationNegativePrompt?: string;
  lastGenerationProvenance: PromptProvenance | undefined;
}): PromptHistoryState {
  if (input.stored !== undefined && input.stored.history.length > 0) {
    return input.stored;
  }
  if (input.lastGenerationPrompt === input.prompt) {
    return historyForRestoredPrompt(
      input.prompt,
      input.lastGenerationProvenance,
      input.negativePrompt,
    );
  }
  return input.stored ?? { history: [], index: -1 };
}
