import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  MAX_GENERATION_SEED,
  clampGenerationSeed,
  parseGenerationSeedInput,
  randomGenerationSeed,
  shouldRollSeedAfterSuccess,
} from "./generation-seed.ts";

describe("generationSeed", () => {
  it("clamps to the inclusive 0..max range", () => {
    assert.equal(clampGenerationSeed(-4), 0);
    assert.equal(clampGenerationSeed(12.9), 12);
    assert.equal(clampGenerationSeed(MAX_GENERATION_SEED + 10), MAX_GENERATION_SEED);
    assert.equal(clampGenerationSeed(Number.NaN), 0);
  });

  it("parses whole integers and rejects mixed or decimal text", () => {
    assert.equal(parseGenerationSeedInput("42"), 42);
    assert.equal(parseGenerationSeedInput("  12  "), 12);
    assert.equal(parseGenerationSeedInput("-4"), 0);
    assert.equal(parseGenerationSeedInput("12.9"), null);
    assert.equal(parseGenerationSeedInput("7abc9"), null);
    assert.equal(parseGenerationSeedInput(""), null);
    assert.equal(parseGenerationSeedInput("abc"), null);
    assert.equal(parseGenerationSeedInput("00000000000"), 0);
    assert.equal(parseGenerationSeedInput("1" + "0".repeat(309)), MAX_GENERATION_SEED);
    assert.equal(parseGenerationSeedInput("-" + "9".repeat(400)), 0);
    assert.equal(parseGenerationSeedInput(String(MAX_GENERATION_SEED + 1)), MAX_GENERATION_SEED);
  });

  it("rolls after success only when unlocked and the seed is unchanged", () => {
    assert.equal(shouldRollSeedAfterSuccess({ seed: 7, locked: false }, 7), true);
    assert.equal(shouldRollSeedAfterSuccess({ seed: 7, locked: true }, 7), false);
    assert.equal(shouldRollSeedAfterSuccess({ seed: 7, locked: false }, 8), false);
  });

  it("randomGenerationSeed never returns the excluded seed", () => {
    const original = Math.random;
    Math.random = () => 7 / (MAX_GENERATION_SEED + 1);
    try {
      assert.notEqual(randomGenerationSeed(7), 7);
    } finally {
      Math.random = original;
    }
  });
});
