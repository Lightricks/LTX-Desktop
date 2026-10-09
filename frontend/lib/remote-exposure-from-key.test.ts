import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { remoteExposureFromKey } from "./remote-exposure-from-key.ts";

describe("remoteExposureFromKey", () => {
  it("maps Home and End to Off and On", () => {
    assert.equal(remoteExposureFromKey(true, "Home"), false);
    assert.equal(remoteExposureFromKey(false, "Home"), false);
    assert.equal(remoteExposureFromKey(true, "End"), true);
    assert.equal(remoteExposureFromKey(false, "End"), true);
  });

  it("wraps arrow keys between Off and On", () => {
    assert.equal(remoteExposureFromKey(false, "ArrowRight"), true);
    assert.equal(remoteExposureFromKey(true, "ArrowRight"), false);
    assert.equal(remoteExposureFromKey(true, "ArrowLeft"), false);
    assert.equal(remoteExposureFromKey(false, "ArrowLeft"), true);
    assert.equal(remoteExposureFromKey(false, "ArrowDown"), true);
    assert.equal(remoteExposureFromKey(true, "ArrowUp"), false);
  });

  it("ignores unrelated keys", () => {
    assert.equal(remoteExposureFromKey(true, "Enter"), null);
    assert.equal(remoteExposureFromKey(false, " "), null);
    assert.equal(remoteExposureFromKey(true, "Tab"), null);
  });
});
