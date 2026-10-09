import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  videoFilmstripFrameCount,
  videoFilmstripTimes,
} from "./filmstripLayout.ts";

describe("videoFilmstripFrameCount", () => {
  it("keeps a readable minimum on a narrow track", () => {
    assert.equal(videoFilmstripFrameCount(40), 6);
  });

  it("scales with track width and caps the tile count", () => {
    assert.equal(videoFilmstripFrameCount(360), 10);
    assert.equal(videoFilmstripFrameCount(2000), 24);
  });
});

describe("videoFilmstripTimes", () => {
  it("returns no samples for an empty clip", () => {
    assert.deepEqual(videoFilmstripTimes(0, 8), []);
    assert.deepEqual(videoFilmstripTimes(8, 0), []);
  });

  it("samples the middle of each tile inside the clip", () => {
    assert.deepEqual(videoFilmstripTimes(8, 4), [1, 3, 5, 7]);
  });
});
