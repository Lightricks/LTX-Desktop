import { useEffect, useState } from "react";

const NESTED_INTERACTIVE_SELECTOR = [
  "a[href]",
  "button",
  "input",
  "select",
  "textarea",
  "[contenteditable='true']",
  "[role='button']",
].join(",");

export function isNestedInteractiveTarget(
  target: EventTarget | null,
  currentTarget: Element,
): boolean {
  if (!(target instanceof Element)) return false;
  if (target instanceof HTMLElement && target.isContentEditable) return true;
  const interactiveTarget = target.closest(NESTED_INTERACTIVE_SELECTOR);
  return interactiveTarget !== null && interactiveTarget !== currentTarget;
}

export function useFinePointer(): boolean {
  const [isFinePointer, setIsFinePointer] = useState(() => {
    return typeof window !== "undefined" &&
      typeof window.matchMedia === "function"
      ? window.matchMedia("(hover: hover) and (pointer: fine)").matches
      : false;
  });

  useEffect(() => {
    const query = window.matchMedia("(hover: hover) and (pointer: fine)");
    const sync = () => setIsFinePointer(query.matches);
    sync();
    query.addEventListener("change", sync);
    return () => query.removeEventListener("change", sync);
  }, []);

  return isFinePointer;
}
