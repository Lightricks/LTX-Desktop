import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { failureLabel, fixHint, formatFootage, renderBand } from "./dashboardFormat.ts";

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

describe("failure labels", () => {
  it("labels CUDA_OOM and JOB_TOO_LARGE instead of raw codes", () => {
    assert.notEqual(failureLabel("CUDA_OOM"), "CUDA_OOM");
    assert.notEqual(failureLabel("JOB_TOO_LARGE"), "JOB_TOO_LARGE");
    assert.notEqual(fixHint("CUDA_OOM"), "Generate again.");
    assert.notEqual(fixHint("JOB_TOO_LARGE"), "Generate again.");
  });

  it("falls back to the raw code and a generic hint for unknown codes", () => {
    assert.equal(failureLabel("SOME_FUTURE_CODE"), "SOME_FUTURE_CODE");
    assert.equal(fixHint("SOME_FUTURE_CODE"), "Generate again.");
  });
});
