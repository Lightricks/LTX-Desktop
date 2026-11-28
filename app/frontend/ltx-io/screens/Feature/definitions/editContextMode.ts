import { EDIT_CONTEXT_LABELS } from "../../../lib/extendLabels.ts";
import type { EditContextMode } from "../../../lib/editContextMode.ts";
import type { FieldOption } from "../types.ts";

export const EDIT_CONTEXT_MODE_OPTIONS: FieldOption<EditContextMode>[] = [
  { value: "long", label: EDIT_CONTEXT_LABELS.long },
  { value: "short", label: EDIT_CONTEXT_LABELS.short },
];

/** The tooltip of the Context field. ``shortScope`` names where Short keeps its 2 seconds. */
export function contextModeTooltip(shortScope: string): string {
  return (
    "Long: uses as much of the clip as fits as context. Most consistent, slower.\n\n" +
    `Short: uses only the 2 seconds ${shortScope} as context. ` +
    "Faster, but may drift from the rest of the clip."
  );
}
