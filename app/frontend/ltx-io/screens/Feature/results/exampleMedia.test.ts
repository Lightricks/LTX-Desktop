import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { exampleEmptyPlayback } from "./exampleMedia.ts";

describe("exampleEmptyPlayback", () => {
  it("falls back to the poster when the browser cannot decode the example WebM", () => {
    assert.equal(
      exampleEmptyPlayback("/assets/example-a1b2c3.webm", () => ""),
      "poster",
    );
    assert.equal(
      exampleEmptyPlayback("example.webm?v=1", () => ""),
      "poster",
    );
  });

  it("keeps click-to-play video when the browser can decode WebM", () => {
    assert.equal(
      exampleEmptyPlayback("/assets/example-a1b2c3.webm", () => "maybe"),
      "video",
    );
    assert.equal(
      exampleEmptyPlayback("/assets/example-a1b2c3.webm", () => "probably"),
      "video",
    );
  });

  it("applies the same canPlayType gate to mp4 example URLs", () => {
    assert.equal(
      exampleEmptyPlayback("/assets/example.mp4", () => "probably"),
      "video",
    );
    assert.equal(
      exampleEmptyPlayback("/assets/example.mp4", () => ""),
      "poster",
    );
  });
});
