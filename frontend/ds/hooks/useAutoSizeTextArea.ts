import { useEffect } from "react";

/**
 * Auto-sizes a textarea element to fit its content by adjusting
 * its height on every value change.
 */
export function useAutoSizeTextArea(
  textAreaRef: React.RefObject<HTMLTextAreaElement | null>,
  value: string,
) {
  useEffect(() => {
    if (textAreaRef.current) {
      textAreaRef.current.style.height = "0px";
      const scrollHeight = textAreaRef.current.scrollHeight;
      textAreaRef.current.style.height = scrollHeight + "px";
    }
  }, [textAreaRef, value]);
}
