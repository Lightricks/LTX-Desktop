import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { resolveShowManualEnhance } from "./manualEnhanceVisibility.ts";

describe("resolveShowManualEnhance", () => {
  it("hides Enhance when Desktop auto-enhance is on", () => {
    assert.equal(resolveShowManualEnhance(true, true), false);
  });

  it("shows Enhance when Desktop auto-enhance is off", () => {
    assert.equal(resolveShowManualEnhance(false, false), true);
  });

  it("follows status fetched on feature enter when Remote has no live setting", () => {
    assert.equal(resolveShowManualEnhance(undefined, false), false);
    assert.equal(resolveShowManualEnhance(undefined, true), true);
  });
});
