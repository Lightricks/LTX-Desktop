import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { createRowHeightEstimator } from "./rowHeightEstimator.ts";

describe("createRowHeightEstimator", () => {
  it("replaces the value of a row that is measured again, so it does not count twice", () => {
    const estimator = createRowHeightEstimator();
    const width = 160; // The frame is 90 px high.

    // Row "a" has 40 px of overhead, row "b" has 60 px: the average is 50.
    estimator.record("a", 130, width);
    estimator.record("b", 150, width);
    assert.equal(estimator.estimate(width), 90 + 50);

    // Row "a" grows to 80 px (a negative prompt line). The average is (80 + 60) / 2 = 70.
    // A double count would give (40 + 80 + 60) / 3 = 60, or a drifting sum.
    estimator.record("a", 170, width);
    estimator.record("a", 170, width);
    assert.equal(estimator.estimate(width), 90 + 70);
  });
});
