import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  hasNewSucceededGeneration,
  isInFlightGeneration,
  readGenerationParams,
  type Generation,
} from "./resultsFeedModel.ts";

function gen(id: string, status: Generation["status"]): Generation {
  return { id, status } as Generation;
}

describe("isInFlightGeneration", () => {
  it("treats queued, running, and cancelling as in-flight", () => {
    assert.equal(isInFlightGeneration(gen("a", "queued")), true);
    assert.equal(isInFlightGeneration(gen("b", "running")), true);
    assert.equal(isInFlightGeneration(gen("cancelling", "cancelling")), true);
    assert.equal(isInFlightGeneration(gen("c", "succeeded")), false);
    assert.equal(isInFlightGeneration(gen("d", "failed")), false);
    assert.equal(isInFlightGeneration(gen("e", "cancelled")), false);
  });
});

describe("hasNewSucceededGeneration", () => {
  it("ignores the first snapshot so hydrating Results does not refresh Assets", () => {
    assert.equal(hasNewSucceededGeneration(undefined, [gen("a", "succeeded")]), false);
  });

  it("detects a job that reaches succeeded after an in-flight snapshot", () => {
    assert.equal(
      hasNewSucceededGeneration([gen("a", "running")], [gen("a", "succeeded")]),
      true,
    );
  });

  it("does not treat an unchanged succeeded list as a new output", () => {
    assert.equal(
      hasNewSucceededGeneration([gen("a", "succeeded")], [gen("a", "succeeded")]),
      false,
    );
  });
});

describe("readGenerationParams", () => {
  it("returns the params object when present", () => {
    const params = { prompt: "hi", fps: 24 };
    assert.deepEqual(readGenerationParams({ params }), params);
  });

  it("returns null for non-object specs", () => {
    assert.equal(readGenerationParams(null), null);
    assert.equal(readGenerationParams(undefined), null);
    assert.equal(readGenerationParams("nope"), null);
  });

  it("returns null when params is missing or not a plain object", () => {
    assert.equal(readGenerationParams({}), null);
    assert.equal(readGenerationParams({ params: null }), null);
    assert.equal(readGenerationParams({ params: [1, 2, 3] }), null);
  });
});
