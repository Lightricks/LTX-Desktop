import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { calculateBarData, calculateBarDataSymmetric } from "./utils.ts";

describe("audio visualizer bar data", () => {
  it("averages the usable low-frequency bins into one bar per slot", () => {
    const frequencyData = new Uint8Array(10);
    frequencyData.set([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);

    assert.deepEqual(calculateBarData(frequencyData, 12, 2, 4, 0.4), [15, 35]);
  });

  it("mirrors the left-half bars onto the right", () => {
    const frequencyData = new Uint8Array(10);
    frequencyData.set([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);

    assert.deepEqual(
      calculateBarDataSymmetric(frequencyData, 24, 2, 4, 0.4),
      [15, 35, 35, 15],
    );
  });

  it("returns no bars when the canvas cannot fit a mirrored pair", () => {
    assert.deepEqual(
      calculateBarDataSymmetric(new Uint8Array(10), 4, 2, 4),
      [],
    );
  });
});
