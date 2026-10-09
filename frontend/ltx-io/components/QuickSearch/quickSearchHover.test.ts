import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { quickSearchHoverIntent } from "./quickSearchHover.ts";

const origin = { x: 10, y: 20 };

describe("quickSearchHoverIntent", () => {
  it("switches immediately on the first move onto a row", () => {
    assert.equal(
      quickSearchHoverIntent({ x: Number.NaN, y: Number.NaN }, origin),
      "immediate",
    );
  });

  it("switches immediately when the pointer moves down the list", () => {
    assert.equal(quickSearchHoverIntent(origin, { x: 12, y: 48 }), "immediate");
  });

  it("defers when the pointer moves right toward the preview", () => {
    assert.equal(quickSearchHoverIntent(origin, { x: 40, y: 24 }), "defer");
  });

  it("ignores a pointer event that did not move", () => {
    assert.equal(quickSearchHoverIntent(origin, origin), "ignore");
  });
});
