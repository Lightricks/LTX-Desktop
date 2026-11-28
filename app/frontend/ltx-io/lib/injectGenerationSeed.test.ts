import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { injectGenerationSeed } from "./injectGenerationSeed.ts";

describe("injectGenerationSeed", () => {
  it("sets params.seed and leaves other params in place", () => {
    const body = injectGenerationSeed(
      {
        contract_version: 1,
        params: { prompt: "a fox", model: "ltx-2.5-fast" },
      },
      42,
    );

    assert.deepEqual(body, {
      contract_version: 1,
      params: { prompt: "a fox", model: "ltx-2.5-fast", seed: 42 },
    });
  });

  it("still pins seed 0", () => {
    assert.deepEqual(injectGenerationSeed({ params: {} }, 0), { params: { seed: 0 } });
  });
});
