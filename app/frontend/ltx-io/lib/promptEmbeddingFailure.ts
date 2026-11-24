/** Queued-job code when LTX API prompt embedding fails. See generation error_code. */
export const LTX_API_PROMPT_EMBEDDING_FAILED = "LTX_API_PROMPT_EMBEDDING_FAILED";

export const PROMPT_EMBEDDING_FAILED_MESSAGE = "LTX API text encoding failed.";

/** Phone copy: Remote has no Settings modal, so the fix happens on the desktop app. */
export const PROMPT_EMBEDDING_FAILED_REMOTE_MESSAGE =
  "LTX API text encoding failed. Update it in Settings on the desktop app.";

export const OPEN_TEXT_ENCODING_LABEL = "Text encoding";

export type PromptEmbeddingFailureNotice = {
  body: string;
  /** Set only on desktop, where the button can open Settings. */
  openSettingsLabel?: string;
};

/**
 * Explain a failed LTX API prompt embedding on an Explore result.
 *
 * The settings shortcut is only for the desktop app, and only while Text encoding
 * is still LTX API. A job can keep this error code after the user switches to the
 * local encoder, and the button would send them to a choice they already made.
 *
 * Remote cannot read that setting (GET /api/settings is not on the phone). When the
 * host omits `usesLtxApiTextEncoding`, the phone still explains the code, without
 * a button.
 */
export function promptEmbeddingFailureNotice(input: {
  errorCode: string | null | undefined;
  usesLtxApiTextEncoding: boolean | undefined;
  canOpenSettings: boolean;
}): PromptEmbeddingFailureNotice | null {
  if (input.errorCode !== LTX_API_PROMPT_EMBEDDING_FAILED) return null;
  if (input.usesLtxApiTextEncoding === false) return null;

  if (input.canOpenSettings) {
    if (input.usesLtxApiTextEncoding !== true) return null;
    return {
      body: PROMPT_EMBEDDING_FAILED_MESSAGE,
      openSettingsLabel: OPEN_TEXT_ENCODING_LABEL,
    };
  }

  return { body: PROMPT_EMBEDDING_FAILED_REMOTE_MESSAGE };
}
