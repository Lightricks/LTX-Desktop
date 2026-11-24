import { budgetTooltip, type BudgetReadout } from "./budget-figure.ts";

/** Budget warning tiers. The backend decides admission and tier; this module only renders them. */

export type BudgetMode = "brave" | "cautious";

export type WarningTone = "caveat" | "risk";

/** Tight is a passive note. Over and unmeasured are the bets Brave asks about and Cautious refuses. */
export type SelectionTier = "tight" | "over" | "unmeasured";

export type ConfirmableTier = Exclude<SelectionTier, "tight">;

export type SelectionRisk = {
  tier: SelectionTier;
  message: string;
  readout?: BudgetReadout | null;
};

const TIER_RANK: Record<SelectionTier, number> = {
  tight: 1,
  over: 2,
  unmeasured: 3,
};

/** The backend only emits the two names; anything else is the default. */
export function readBudgetMode(value: string | null | undefined): BudgetMode {
  return value === "cautious" ? "cautious" : "brave";
}

export type PreflightCellStatus =
  | { status: "clear" }
  | {
      status: "warned";
      message: string | null;
      tier: ConfirmableTier;
      readout: BudgetReadout | null;
    }
  | { status: "blocked"; message: string | null; readout: BudgetReadout | null };

/** `ok` already applies the echoed mode. An ok cell with a bet tier is warned. */
export function preflightCellStatus(
  cell:
    | {
        ok: boolean;
        message?: string | null;
        tier?: ConfirmableTier | null;
        readout?: BudgetReadout | null;
      }
    | null
    | undefined,
): PreflightCellStatus {
  if (cell == null) return { status: "clear" };
  const message = cell.message ?? null;
  const readout = cell.readout ?? null;
  if (!cell.ok) return { status: "blocked", message, readout };
  if (isConfirmableBudgetTier(cell.tier)) {
    return { status: "warned", message, tier: cell.tier, readout };
  }
  return { status: "clear" };
}

/** Over and unmeasured ask before generate. Tight stays on the option icon. */
export function isConfirmableBudgetTier(
  tier: string | null | undefined,
): tier is ConfirmableTier {
  return tier === "over" || tier === "unmeasured";
}

/** The issue Generate confirms before it runs, or null when nothing needs asking. */
export function confirmableBudgetIssue<T extends { tier?: string | null }>(
  issues: readonly T[],
): (T & { tier: ConfirmableTier }) | null {
  return (
    issues.find((issue): issue is T & { tier: ConfirmableTier } =>
      isConfirmableBudgetTier(issue.tier),
    ) ?? null
  );
}

export function worseRisk(
  current: SelectionRisk | null,
  next: SelectionRisk | null,
): SelectionRisk | null {
  if (current == null) return next;
  if (next == null) return current;
  return TIER_RANK[next.tier] > TIER_RANK[current.tier] ? next : current;
}

/** Risk outranks a quality caveat. One icon, risk line first. */
export function riskOptionFields(
  risk: string | undefined,
  caveat: string | undefined,
): { warning?: string; warningTone?: WarningTone } {
  if (risk != null && caveat != null) {
    return { warning: `${risk} ${caveat}`, warningTone: "risk" };
  }
  if (risk != null) return { warning: risk, warningTone: "risk" };
  if (caveat != null) return { warning: caveat };
  return {};
}

export const BET_FAILURE_BACKSTOP = "This job was started despite a memory warning.";

export const NOTHING_FITS_STRIP =
  "Nothing fits this machine — every option may fail or stall.";

export function riskConfirmCopy(tier: ConfirmableTier): { title: string; body: string } {
  switch (tier) {
    case "over":
      return {
        title: "Generate anyway?",
        body: "Your computer may become very slow or freeze while generating. You can stop it once it starts running.",
      };
    case "unmeasured":
      return {
        title: "Not tested on this machine",
        body: "It may run out of memory or slow to a crawl. You can stop it once it starts running.",
      };
    default: {
      const unhandled: never = tier;
      return unhandled;
    }
  }
}

export function generateHintForTier(
  tier: SelectionTier | null | undefined,
): string | undefined {
  if (tier === "over") return "May exceed your GPU memory.";
  if (tier === "unmeasured") return "Unmeasured on your hardware.";
  return undefined;
}

export function queueRiskTooltip(
  message: string,
  readout?: BudgetReadout | null,
): string {
  const line = readout ? budgetTooltip(readout) : message;
  return `Queued despite a warning. ${line}`;
}

export function withBetBackstop(body: string, ranOnBet: boolean): string {
  if (!ranOnBet) return body;
  return `${body} ${BET_FAILURE_BACKSTOP}`;
}
