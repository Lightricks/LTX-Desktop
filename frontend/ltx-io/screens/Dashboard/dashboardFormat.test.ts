import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { formatFootage, renderBand } from "./dashboardFormat.ts";

describe("formatFootage", () => {
  it("rounds the total before splitting minutes, so a minute boundary is not 60s", () => {
    assert.equal(formatFootage(119.6), "2m");
    assert.equal(formatFootage(59.6), "1m");
    assert.equal(formatFootage(61.2), "1m 1s");
  });
});

describe("renderBand", () => {
  it("keeps exactly three minutes in the middle band", () => {
    assert.equal(renderBand(59_999), "fast");
    assert.equal(renderBand(60_000), "mid");
    assert.equal(renderBand(180_000), "mid");
    assert.equal(renderBand(180_001), "slow");
  });
});
