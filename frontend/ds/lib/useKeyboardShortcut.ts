// Keyboard-shortcut hook for the design system, with `isInputElement` inlined so
// `frontend/ds/` stays self-contained.
import { useEffect } from "react";

/**
 * True for input-like elements where users type text (INPUT, TEXTAREA, contentEditable).
 */
function isInputElement(element: Element | null): boolean {
  if (!element) return false;
  if (element.tagName === "INPUT" || element.tagName === "TEXTAREA") {
    return true;
  }
  if (element instanceof HTMLElement && element.contentEditable === "true") {
    return true;
  }
  return false;
}

interface KeyboardShortcutOptions {
  preventInInputs?: boolean;
}

interface UseKeyboardShortcutProps {
  shouldHandle: (event: KeyboardEvent) => boolean;
  onDown: (event: KeyboardEvent) => void;
  onUp?: (event: KeyboardEvent) => void;
  options?: KeyboardShortcutOptions;
}

export function useKeyboardShortcut({
  shouldHandle,
  onDown,
  onUp,
  options = { preventInInputs: false },
}: UseKeyboardShortcutProps) {
  const { preventInInputs } = options;

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (preventInInputs) {
        const activeElement = document.activeElement;
        if (isInputElement(activeElement)) {
          return;
        }
      }

      if (shouldHandle(event)) {
        onDown(event);
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      if (preventInInputs) {
        const activeElement = document.activeElement;
        if (isInputElement(activeElement)) {
          return;
        }
      }

      if (shouldHandle(event)) {
        onUp?.(event);
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    document.addEventListener("keyup", handleKeyUp);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("keyup", handleKeyUp);
    };
  }, [shouldHandle, onDown, onUp, preventInInputs]);
}
