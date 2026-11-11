import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { generationNegativePromptLine } from "./generationNegativePrompt.ts";

const line = (params: Record<string, unknown>) => generationNegativePromptLine({ params });

describe("generationNegativePromptLine", () => {
  it("shows the trimmed text and the scale when NAG is on, because that is what the model got", () => {
    assert.equal(
      line({ negativePrompt: "  blurry  ", nagScale: 11 }),
      "Negative prompt (scale: 11): blurry",
    );
    assert.equal(
      line({ negativePrompt: "blurry", nagScale: 7.5 }),
      "Negative prompt (scale: 7.5): blurry",
    );
  });

  it("shows alpha and tau after the scale when the row holds them", () => {
    assert.equal(
      line({ negativePrompt: "clouds", nagScale: 11, nagAlpha: 0.5, nagTau: 2.5 }),
      "Negative prompt (scale: 11, α: 0.5, τ: 2.5): clouds",
    );
  });

  it("shows none when the text is empty or only spaces", () => {
    assert.equal(line({ negativePrompt: "" }), "Negative prompt: (none)");
    assert.equal(line({ negativePrompt: "   ", nagScale: 11 }), "Negative prompt: (none)");
  });

  it("does not show typed text as sent when NAG is off", () => {
    // The backend drops nagScale when NAG is off, and keeps 0 when the strength is 0.
    assert.equal(line({ negativePrompt: "blurry" }), "Negative prompt: (none, NAG off)");
    assert.equal(
      line({ negativePrompt: "blurry", nagScale: 0 }),
      "Negative prompt: (none, NAG off)",
    );
    assert.equal(
      line({ negativePrompt: "blurry", nagScale: null }),
      "Negative prompt: (none, NAG off)",
    );
  });

  it("returns null for a row without a negative prompt field", () => {
    assert.equal(line({ prompt: "a cat" }), null);
    assert.equal(generationNegativePromptLine(undefined), null);
    assert.equal(generationNegativePromptLine({}), null);
  });
});
