import { useEffect, useState } from "react";

const BASE_TYPING_SPEED_MS = 40;
const DELETE_SPEED_MS = 12;
const INITIAL_TYPING_DELAY_MS = 100;
const HOLD_AFTER_TYPING_MS = 3500;
const PUNCTUATION_DELAY_FACTOR = 3;
const SPACE_DELAY_FACTOR = 1.2;

function getDelayAfterCharacter(char: string): number {
  switch (char) {
    case ",":
    case ".":
    case "!":
    case "?":
      return BASE_TYPING_SPEED_MS * PUNCTUATION_DELAY_FACTOR;
    case " ":
      return BASE_TYPING_SPEED_MS * SPACE_DELAY_FACTOR;
    default:
      return BASE_TYPING_SPEED_MS;
  }
}

/**
 * Cycles through placeholder strings: type out, hold, delete fast, next.
 * Callers should pass a stable non-empty array (module constant or useMemo).
 */
export function useTypewriterPlaceholder(placeholders: string[], enabled = true): string {
  const [displayedText, setDisplayedText] = useState("");

  const [cycle, setCycle] = useState(0);
  const currentPlaceholder = placeholders[cycle % placeholders.length];

  useEffect(() => {
    if (!enabled) return;

    let charIndex = 0;
    let timeout: ReturnType<typeof setTimeout>;

    const deleteCharacters = () => {
      if (charIndex <= 0) {
        setCycle((prev) => prev + 1);
        return;
      }
      charIndex -= 1;
      setDisplayedText(currentPlaceholder.slice(0, charIndex));
      timeout = setTimeout(deleteCharacters, DELETE_SPEED_MS);
    };

    const typeNextCharacter = () => {
      if (charIndex >= currentPlaceholder.length) {
        timeout = setTimeout(deleteCharacters, HOLD_AFTER_TYPING_MS);
        return;
      }
      setDisplayedText(currentPlaceholder.slice(0, charIndex + 1));
      timeout = setTimeout(
        typeNextCharacter,
        getDelayAfterCharacter(currentPlaceholder[charIndex]),
      );
      charIndex += 1;
    };

    timeout = setTimeout(typeNextCharacter, INITIAL_TYPING_DELAY_MS);

    return () => {
      clearTimeout(timeout);
    };
  }, [currentPlaceholder, cycle, enabled]);

  return enabled ? displayedText : "";
}
