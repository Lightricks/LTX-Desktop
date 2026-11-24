import type { components } from "../generated/backend-openapi.ts";

/** Need and have, decided where the budget compared them. */
export type BudgetReadout = components["schemas"]["BudgetReadout"];

const LABELS: Record<BudgetReadout["kind"], string> = {
  gpu: "GPU",
  gpu_busy: "GPU busy",
  mac: "Mac",
  ram: "RAM",
  tight: "Tight",
  untested: "Untested",
};

function gb(value: number, whole: boolean): string {
  if (whole) return String(Math.round(value));
  const rounded = Math.round(value * 10) / 10;
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
}

export function budgetLabel(readout: BudgetReadout): string {
  return LABELS[readout.kind];
}

export function budgetNumbers(
  readout: BudgetReadout,
): { need: string; have: string } | null {
  if (readout.need_gb == null || readout.have_gb == null) return null;
  const whole = readout.kind === "ram";
  return {
    need: gb(readout.need_gb, whole),
    have: gb(readout.have_gb, whole),
  };
}

/**
 * The badge tooltip: the budget line, then the option's own note when it has one. The badge
 * replaces the option's warning icon, so a note the budget line does not cover goes here.
 */
export function budgetTooltip(readout: BudgetReadout, note?: string): string {
  const reason = budgetReason(readout);
  return note ? `${reason} ${note}` : reason;
}

/** One tooltip line. The pair itself is the badge. */
function budgetReason(readout: BudgetReadout): string {
  const numbers = budgetNumbers(readout);
  if (numbers == null) {
    return readout.kind === "untested"
      ? "Not tested on this hardware."
      : "Close to the limit. It should still run.";
  }
  const { need, have } = numbers;
  switch (readout.kind) {
    case "gpu":
      return `Needs ${need} GB. This GPU has ${have} GB.`;
    case "gpu_busy":
      return `Needs ${need} GB. ${have} GB is free right now.`;
    case "mac":
      return `Needs ${need} GB. This Mac allows ${have} GB.`;
    case "ram":
      return `Needs ${need} GB of memory. This computer has ${have} GB.`;
    case "tight":
      return "Close to the limit. It should still run.";
    case "untested":
      return "Not tested on this hardware.";
    default: {
      const _exhaustive: never = readout.kind;
      return _exhaustive;
    }
  }
}

export function budgetIsStressed(readout: BudgetReadout): boolean {
  return readout.kind !== "tight" && readout.kind !== "untested";
}
