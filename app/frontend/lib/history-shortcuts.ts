export type HistoryShortcut = "back" | "forward";

/**
 * Electron has no browser chrome, so these keys are handled in the page.
 * Ctrl+Arrow stays a text-field word jump and is never history.
 */
export function historyShortcutAction(input: {
  platform: string | undefined;
  code: string;
  altKey: boolean;
  metaKey: boolean;
  ctrlKey: boolean;
  shiftKey: boolean;
  inTextEntry: boolean;
  modalOpen: boolean;
  canGoBack: boolean;
  canGoForward: boolean;
}): HistoryShortcut | null {
  if (input.inTextEntry || input.shiftKey || input.modalOpen) return null;

  const mac = input.platform === "darwin";
  const back = mac
    ? input.metaKey &&
      !input.ctrlKey &&
      !input.altKey &&
      (input.code === "BracketLeft" || input.code === "ArrowLeft")
    : input.altKey && !input.metaKey && !input.ctrlKey && input.code === "ArrowLeft";
  const forward = mac
    ? input.metaKey &&
      !input.ctrlKey &&
      !input.altKey &&
      (input.code === "BracketRight" || input.code === "ArrowRight")
    : input.altKey && !input.metaKey && !input.ctrlKey && input.code === "ArrowRight";

  if (back && input.canGoBack) return "back";
  if (forward && input.canGoForward) return "forward";
  return null;
}
