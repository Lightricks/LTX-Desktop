import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  icLoraFpsOptions,
  icLoraMaxInputSeconds,
  icLoraOriginalFps,
} from "./icLoraFps.ts";

function labels(sourceFps: number): string[] {
  return icLoraFpsOptions(sourceFps).map((option) => option.label);
}

describe("ic-lora fps options", () => {
  it("labels the closest rate that is not above the source as Original", () => {
    assert.deepEqual(labels(60), ["24", "25", "48", "50 (Original)"]);
    assert.deepEqual(labels(30), ["24", "25 (Original)"]);
    assert.deepEqual(labels(29.97), ["24", "25 (Original)"]);
    assert.deepEqual(labels(25), ["24", "25 (Original)"]);
    assert.deepEqual(labels(24), ["24 (Original)"]);
    assert.deepEqual(labels(23.976), ["24 (Original)"]);
    assert.deepEqual(labels(15), ["15 (Original)"]);
  });

  it("picks the lower rate when two supported rates are equally close", () => {
    assert.equal(icLoraOriginalFps(24.5), 24);
    assert.equal(icLoraOriginalFps(49), 48);
    assert.deepEqual(labels(49), ["24", "25", "48 (Original)"]);
  });

  it("shortens the trim cap as the selected rate rises", () => {
    assert.equal(icLoraMaxInputSeconds(24), 10);
    assert.equal(icLoraMaxInputSeconds(25), 9.6);
    assert.equal(icLoraMaxInputSeconds(48), 5);
    assert.equal(icLoraMaxInputSeconds(50), 4.8);
    assert.equal(icLoraMaxInputSeconds(15), 10);
  });
});
