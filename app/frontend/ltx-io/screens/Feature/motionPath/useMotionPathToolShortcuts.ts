import { useEffect, useRef, useState } from "react";

import type { MotionPathTool } from "./MotionPathCanvas";

export type UseMotionPathToolShortcutsOptions = {
  enabled: boolean;
  selectedTool: MotionPathTool;
  setSelectedTool: (tool: MotionPathTool) => void;
};

export type UseMotionPathToolShortcutsResult = {
  activeTool: MotionPathTool;
  isCommandEraseHeld: boolean;
  isOptionConvertHeld: boolean;
};

export function useMotionPathToolShortcuts({
  enabled,
  selectedTool,
  setSelectedTool,
}: UseMotionPathToolShortcutsOptions): UseMotionPathToolShortcutsResult {
  const [isCommandEraseHeld, setIsCommandEraseHeld] = useState(false);
  const isCommandEraseHeldRef = useRef(false);
  const [isOptionConvertHeld, setIsOptionConvertHeld] = useState(false);
  const isOptionConvertHeldRef = useRef(false);

  const activeTool: MotionPathTool = isCommandEraseHeld
    ? "erase"
    : isOptionConvertHeld
      ? "convert"
      : selectedTool;

  useEffect(() => {
    const isCommandKey = (event: KeyboardEvent) =>
      event.key === "Meta" || event.key === "Control";
    const isOptionKey = (event: KeyboardEvent) => event.key === "Alt";

    const releaseCommandHold = () => {
      if (!isCommandEraseHeldRef.current) return;
      isCommandEraseHeldRef.current = false;
      setIsCommandEraseHeld(false);
    };

    const releaseOptionHold = () => {
      if (!isOptionConvertHeldRef.current) return;
      isOptionConvertHeldRef.current = false;
      setIsOptionConvertHeld(false);
    };

    const releaseAllHolds = () => {
      releaseCommandHold();
      releaseOptionHold();
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.repeat || !enabled) return;

      const target = event.target;
      if (
        target instanceof HTMLElement &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)
      ) {
        return;
      }

      if (isCommandKey(event) && !isCommandEraseHeldRef.current) {
        isCommandEraseHeldRef.current = true;
        setIsCommandEraseHeld(true);
        return;
      }
      if (isOptionKey(event) && !isOptionConvertHeldRef.current) {
        if (event.metaKey || event.ctrlKey) return;
        isOptionConvertHeldRef.current = true;
        setIsOptionConvertHeld(true);
        return;
      }

      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const key = event.key.toLowerCase();
      if (key === "b") {
        event.preventDefault();
        setSelectedTool("freehand");
      } else if (key === "p") {
        event.preventDefault();
        setSelectedTool("pen");
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      if (isCommandKey(event)) releaseCommandHold();
      if (isOptionKey(event)) releaseOptionHold();
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", releaseAllHolds);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", releaseAllHolds);
      releaseAllHolds();
    };
  }, [enabled, setSelectedTool]);

  return { activeTool, isCommandEraseHeld, isOptionConvertHeld };
}
