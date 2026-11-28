import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  DEFAULT_NAG_STRENGTH,
  injectNagStrength,
  nagStrengthSummary,
  type NagStrength,
} from "./nagStrength.ts";

const body = (negativePrompt: unknown) => ({ params: { prompt: "a fox", negativePrompt } });
const strength = (preset: NagStrength["preset"], custom = DEFAULT_NAG_STRENGTH.custom): NagStrength => ({
  preset,
  custom,
});

describe("injectNagStrength", () => {
  it("sends nothing for the default preset, so the run keeps the Settings scale", () => {
    const input = body("clouds");
    assert.equal(injectNagStrength(input, DEFAULT_NAG_STRENGTH), input);
  });

  it("sends only alpha for Strong", () => {
    assert.deepEqual(injectNagStrength(body("clouds"), strength("strong")).params, {
      prompt: "a fox",
      negativePrompt: "clouds",
      nagAlpha: 0.5,
    });
  });

  it("sends only alpha for Max", () => {
    assert.deepEqual(injectNagStrength(body("clouds"), strength("max")).params, {
      prompt: "a fox",
      negativePrompt: "clouds",
      nagAlpha: 0.65,
    });
  });

  it("leaves the body alone when the negative text is empty, so no knobs go out with NAG off", () => {
    for (const text of ["", "  \n", undefined]) {
      const input = body(text);
      assert.equal(injectNagStrength(input, strength("max")), input);
    }
  });

  it("clamps Custom to the backend limits, so the request is not rejected with a 422", () => {
    const { params } = injectNagStrength(
      body("clouds"),
      strength("custom", { scale: 500, alpha: 3, tau: Number.NaN }),
    );
    assert.deepEqual(
      { scale: params.nagScale, alpha: params.nagAlpha, tau: params.nagTau },
      { scale: 30, alpha: 0.7, tau: 1 },
    );
  });

  it("does not change the input body", () => {
    const input = body("clouds");
    injectNagStrength(input, strength("max"));
    assert.deepEqual(input.params, { prompt: "a fox", negativePrompt: "clouds" });
  });
});

describe("nagStrengthSummary", () => {
  it("names the preset, and spells out the Custom values", () => {
    assert.equal(nagStrengthSummary(strength("strong")), "Strong");
    assert.equal(
      nagStrengthSummary(strength("custom", { scale: 11, alpha: 0.4, tau: 2.5 })),
      "Custom · scale 11, α 0.4, τ 2.5",
    );
  });
});
