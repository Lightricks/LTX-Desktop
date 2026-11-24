import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { genSpaceImageAspectRatio } from "./genspace-image-aspect.ts";

describe("genSpaceImageAspectRatio", () => {
  it("keeps the image picker ratios", () => {
    assert.equal(genSpaceImageAspectRatio("16:9"), "16:9");
    assert.equal(genSpaceImageAspectRatio("1:1"), "1:1");
    assert.equal(genSpaceImageAspectRatio("9:16"), "9:16");
  });

  it("falls back when the shared control holds a video-only ratio", () => {
    assert.equal(genSpaceImageAspectRatio("3:2"), "16:9");
    assert.equal(genSpaceImageAspectRatio("4:3"), "16:9");
    assert.equal(genSpaceImageAspectRatio("4:5"), "16:9");
    assert.equal(genSpaceImageAspectRatio("21:9"), "16:9");
    assert.equal(genSpaceImageAspectRatio(undefined), "16:9");
  });
});
