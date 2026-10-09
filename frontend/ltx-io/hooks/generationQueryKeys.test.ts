import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  generationQueryKeys,
  VIDEO_GENERATION_MODEL_SPECS_QUERY_OPTIONS,
} from "./generationQueryKeys.ts";

describe("generationQueryKeys.specs", () => {
  it("changes when the Desktop modelsVersion changes", () => {
    assert.deepEqual(generationQueryKeys.specs(1), [
      "video-generation-model-specs",
      1,
    ]);
    assert.deepEqual(generationQueryKeys.specs(2), [
      "video-generation-model-specs",
      2,
    ]);
    assert.notDeepEqual(
      generationQueryKeys.specs(1),
      generationQueryKeys.specs(2),
    );
  });

  it("uses a stable Remote key that is not a Desktop version", () => {
    assert.deepEqual(generationQueryKeys.specs(null), [
      "video-generation-model-specs",
      null,
    ]);
    assert.deepEqual(
      generationQueryKeys.specs(null),
      generationQueryKeys.specs(null),
    );
    assert.notDeepEqual(
      generationQueryKeys.specs(null),
      generationQueryKeys.specs(0),
    );
  });
});

describe("video generation model specs query options", () => {
  it("refetches on mount and window focus", () => {
    assert.equal(
      VIDEO_GENERATION_MODEL_SPECS_QUERY_OPTIONS.refetchOnMount,
      true,
    );
    assert.equal(
      VIDEO_GENERATION_MODEL_SPECS_QUERY_OPTIONS.refetchOnWindowFocus,
      true,
    );
  });
});
