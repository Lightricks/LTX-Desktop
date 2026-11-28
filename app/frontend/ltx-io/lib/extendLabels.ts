import { isEditContextMode } from "./editContextMode.ts";

/** Where the new part goes, as the Extend form and the result metadata name it. */
export const EXTEND_MODE_LABELS = {
  start: "Before the clip",
  end: "After the clip",
} as const;

/** How much of the clip the new part continues from. */
export const EDIT_CONTEXT_LABELS = {
  long: "Long",
  short: "Short",
} as const;

export function isExtendMode(value: unknown): value is keyof typeof EXTEND_MODE_LABELS {
  return value === "start" || value === "end";
}

export function extendModeLabel(value: unknown): string | null {
  return isExtendMode(value) ? EXTEND_MODE_LABELS[value] : null;
}

/** The result metadata names the context ("Short context"); the form field does not. */
export function editContextBadge(value: unknown): string | null {
  return isEditContextMode(value) ? `${EDIT_CONTEXT_LABELS[value]} context` : null;
}
