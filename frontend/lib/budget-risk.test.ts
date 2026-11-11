import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { confirmableBudgetIssue } from "./budget-risk.ts";

describe("confirmableBudgetIssue", () => {
  it("tight warning does not open the risk dialog", () => {
    const tight = { id: "budget-risk", tier: "tight", message: "Close to the limit." };
    const over = { id: "budget-risk", tier: "over", message: "Needs 18 GB." };
    assert.equal(confirmableBudgetIssue([tight]), null);
    assert.equal(confirmableBudgetIssue([tight, over]), over);
  });
});
