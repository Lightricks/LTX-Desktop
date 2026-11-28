export const REMOVE_FROM_HISTORY_LABEL = "Remove from history";

export const RESULT_DELETE_CONFIRM_TITLE =
  "Are you sure you want to remove this result?";
export const RESULT_DELETE_CONFIRM_DESCRIPTION =
  "This removes the result from history. The file stays in Assets.";
export const RESULT_DELETE_WITHOUT_ASSET_CONFIRM_DESCRIPTION =
  "This removes the result from history.";
export const RESULT_DELETE_CONFIRM_BUTTON = "Remove result";

export function resultDeleteConfirmation(hasAsset: boolean) {
  return {
    confirmationName: "result-delete-confirm",
    title: RESULT_DELETE_CONFIRM_TITLE,
    description: hasAsset
      ? RESULT_DELETE_CONFIRM_DESCRIPTION
      : RESULT_DELETE_WITHOUT_ASSET_CONFIRM_DESCRIPTION,
    confirmLabel: RESULT_DELETE_CONFIRM_BUTTON,
  };
}

/** A Queue link pointed at a generation that this feature page no longer lists. */
export const RESULT_NOT_FOUND_MESSAGE = "That generation is no longer in this list.";

/** The AlphaGen WebM: the subject with a transparent background. */
export const CUTOUT_ALPHA_FILE_LABEL = "Cutout with alpha (WebM)";
/** The AlphaGen matte MP4: the black and white mask of the subject. */
export const CUTOUT_MATTE_FILE_LABEL = "Matte (MP4)";
/** The AlphaGen GIF: the cutout with a transparent background, for apps that read no WebM. */
export const CUTOUT_GIF_FILE_LABEL = "GIF";
