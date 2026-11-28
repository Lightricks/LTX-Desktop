import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { budgetTooltip, type BudgetReadout } from "./budget-figure.ts";

const mac: BudgetReadout = { kind: "mac", need_gb: 40, have_gb: 24 };

describe("budgetTooltip", () => {
  it("is the budget line alone when the option has no note", () => {
    assert.equal(budgetTooltip(mac), "Needs 40 GB. This Mac allows 24 GB.");
  });

  it("adds the option's note after the budget line", () => {
    assert.equal(
      budgetTooltip(mac, "This runs in 3 overlapping time windows."),
      "Needs 40 GB. This Mac allows 24 GB. This runs in 3 overlapping time windows.",
    );
  });
});
