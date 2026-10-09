/** Queued-job code from a rejected LTX key. See generation error_code. */
export const LTX_INVALID_API_KEY = "LTX_INVALID_API_KEY";

export const REJECTED_LTX_API_KEY_MESSAGE = "This LTX API key isn’t valid.";

/** Phone copy: Remote has no Settings modal, so the fix happens on the desktop app. */
export const REJECTED_LTX_API_KEY_REMOTE_MESSAGE =
  "This LTX API key isn’t valid. Update it in Settings on the desktop app.";

export const UPDATE_LTX_API_KEY_LABEL = "Update API key";

export type RejectedLtxApiKeyNotice = {
  body: string;
  /** Set only on desktop, where the button can open Settings. */
  openSettingsLabel?: string;
};

/**
 * Explain a rejected LTX key on an Explore result.
 *
 * The settings shortcut is only for the desktop app, and only while Text encoding
 * is still LTX API. A job can keep this error code after the user switches to the
 * local encoder, and the button would send them to a key they are no longer using.
 *
 * Remote cannot read that setting (GET /api/settings is not on the phone). When the
 * host omits `usesLtxApiTextEncoding`, the phone still explains the code, without
 * a button.
 */
export function rejectedLtxApiKeyNotice(input: {
  errorCode: string | null | undefined;
  usesLtxApiTextEncoding: boolean | undefined;
  canOpenSettings: boolean;
}): RejectedLtxApiKeyNotice | null {
  if (input.errorCode !== LTX_INVALID_API_KEY) return null;
  if (input.usesLtxApiTextEncoding === false) return null;

  if (input.canOpenSettings) {
    if (input.usesLtxApiTextEncoding !== true) return null;
    return {
      body: REJECTED_LTX_API_KEY_MESSAGE,
      openSettingsLabel: UPDATE_LTX_API_KEY_LABEL,
    };
  }

  return { body: REJECTED_LTX_API_KEY_REMOTE_MESSAGE };
}
