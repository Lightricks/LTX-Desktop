import type { components } from "../../generated/backend-openapi";

/** How much of the clip an Extend or Retake keeps as context. Both features share one set. */
export type EditContextMode = components["schemas"]["ExtendParams"]["contextMode"];

/** What the form, a restored run and the preflight use when no mode is stated. */
export const DEFAULT_EDIT_CONTEXT_MODE: EditContextMode = "long";

export function isEditContextMode(value: unknown): value is EditContextMode {
  return value === "long" || value === "short";
}
