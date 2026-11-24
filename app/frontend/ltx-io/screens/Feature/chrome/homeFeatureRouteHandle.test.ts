import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { selectHomeFeatureId } from "./homeFeatureRouteHandle.ts";

describe("selectHomeFeatureId", () => {
  it("returns the deepest home feature handle", () => {
    assert.equal(
      selectHomeFeatureId([
        { handle: { homeFeatureId: "text-to-video" } },
        { handle: { homeFeatureId: "image-to-video" } },
      ]),
      "image-to-video",
    );
  });

  it("skips missing and malformed handles", () => {
    assert.equal(selectHomeFeatureId([]), null);
    assert.equal(selectHomeFeatureId([{ handle: undefined }]), null);
    assert.equal(selectHomeFeatureId([{ handle: "nope" }]), null);
    assert.equal(
      selectHomeFeatureId([{ handle: { homeFeatureId: "not-a-feature" } }]),
      null,
    );
  });
});
