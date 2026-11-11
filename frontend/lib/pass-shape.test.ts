import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { specFixture } from "../ltx-io/screens/Feature/definitions/specFixture.ts";
import { passShapeChoice } from "./pass-shape.ts";
import type {
  FastCellAsk,
  JobMemoryAsk,
  VideoGenerationModelSpecsResponse,
} from "./video-generation-model-specs.ts";

const ONE_PASS_OVER: JobMemoryAsk = { ask_gib: 22.7, confidence: "solid", windows: 1, tight: false };

/** A 16 GB card: 1080p 2 s fits in one pass inside the measured size, 10 s runs in 3 parts. */
function specs(
  overrides: Partial<VideoGenerationModelSpecsResponse> = {},
): VideoGenerationModelSpecsResponse {
  const base = structuredClone(specFixture());
  const cell = (duration: number, ask: Partial<FastCellAsk>): FastCellAsk => ({
    resolution: "1080p",
    duration,
    fps: 24,
    audio: false,
    confidence: "solid",
    windows: 1,
    tight: false,
    ask_gib: 9.9,
    ...ask,
  });
  const asks = [cell(2, {}), cell(10, { ask_gib: 16, windows: 3, alternative: ONE_PASS_OVER })];
  for (const item of [...base.local_models, ...base.downloaded_local_models]) {
    item.spec.cell_asks = asks;
  }
  return { ...base, memory: { device: "gpu", capacity_gib: 16 }, ...overrides };
}

function at(duration: number) {
  return { model: "ltx-2.5-fast", resolution: "1080p", fps: 24, duration, aspectRatio: "16:9" };
}

describe("passShapeChoice", () => {
  it("the toggle predicate is false when one pass fits inside the fitted range", () => {
    assert.equal(passShapeChoice(specs(), at(2)), null);
    assert.equal(passShapeChoice(specs(), at(10))?.automatic, "parts");
  });

  it("the toggle predicate is false in Cautious", () => {
    assert.equal(passShapeChoice(specs({ budgetMode: "cautious" }), at(10)), null);
  });

  it("the toggle predicate is false while the switch is on", () => {
    assert.equal(passShapeChoice(specs({ neverSplitIntoParts: true }), at(10)), null);
  });
});
