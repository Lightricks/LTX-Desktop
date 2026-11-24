import type { components } from "../generated/backend-openapi";

import type { BudgetReadout } from "./budget-figure.ts";
import { preflightCellStatus } from "./budget-risk.ts";

type CatalogControl = components["schemas"]["IcLoraControl"];
type PreflightCell = components["schemas"]["IcLoraPreflightCell"];

export type IcLoraDurationOption = {
  value: string;
  label: string;
  disabled?: boolean;
  tooltip?: string;
  warning?: string;
  warningTone?: "caveat" | "risk";
  budget?: BudgetReadout | null;
};

export type MergedDurationControl = {
  options: IcLoraDurationOption[];
  /**
   * Set when the control has no selectable option (unsupported machine):
   * render a notice instead of the picker. Null otherwise — even when every
   * option carries an advisory refusal, the picker stays usable.
   */
  emptyMessage: string | null;
};

/**
 * Merge a duration control's kept options with its list-time drops, overlaid
 * by fresh upload preflight verdicts.
 *
 * List-time drops are advisory (the gate previewed a mid-size canvas without
 * the user image, so a dropped duration may still pass on a small source):
 * tooltip, still selectable. Fresh preflight cells decide on real pixels and
 * are authoritative: a failing cell disables with its refusal, a passing cell
 * clears the advisory tooltip. Cells are matched by exact duration; the
 * caller passes null (or omits) while verdicts are missing or stale, and
 * anything unmatched stays as the list-time gate left it.
 */
export function mergeDurationControlOptions(
  control: CatalogControl,
  labelFor: (option: number | string) => string,
  preflightCells: readonly PreflightCell[] | null,
): MergedDurationControl {
  const kept = (control.options ?? []) as (number | string)[];
  const dropped = control.dropped_options ?? [];
  if (kept.length === 0) {
    return {
      options: [],
      emptyMessage: dropped[0]?.message ?? "No durations are available on this machine.",
    };
  }
  const cells = new Map((preflightCells ?? []).map((cell) => [cell.duration, cell]));
  const advisories = new Map(dropped.map((entry) => [entry.value, entry.message]));
  const values = [...kept];
  for (const entry of dropped) {
    if (!values.includes(entry.value)) values.push(entry.value);
  }
  // Numeric controls reconstruct the grid in ascending order; anything else
  // keeps catalog order (a numeric sort over labels would scramble it).
  if (values.every((value): value is number => typeof value === "number")) {
    values.sort((left, right) => left - right);
  }
  return {
    options: values.map((value) => {
      const label = labelFor(value);
      const cell = cells.get(Number(value));
      const status = cell === undefined ? null : preflightCellStatus(cell);
      // GenSpace and the editor have no risk confirm, so a Brave bet disables
      // like a refusal.
      if (status != null && status.status !== "clear") {
        return {
          value: String(value),
          label,
          disabled: true,
          tooltip: status.message ?? undefined,
          ...(status.readout ? { budget: status.readout } : {}),
        };
      }
      // Passing cells clear the advisory; unevaluated values keep it.
      const advisory = cell === undefined ? advisories.get(Number(value)) : undefined;
      return advisory === undefined
        ? { value: String(value), label }
        : { value: String(value), label, tooltip: advisory };
    }),
    emptyMessage: null,
  };
}
