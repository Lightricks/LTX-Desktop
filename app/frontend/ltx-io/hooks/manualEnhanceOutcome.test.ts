import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { MISSING_GEMINI_KEY_MESSAGE } from "../../lib/enhance-gemini-key.ts";
import { enhanceFailureMessage, historyAfterEnhance } from "./manualEnhanceOutcome.ts";

describe("enhance failure message", () => {
  it("another error keeps its own message and does not ask for a key", () => {
    let asked = 0;
    const message = enhanceFailureMessage({ code: "SERVER_ERROR", message: "Boom" }, () => asked++);
    assert.equal(message, "Boom");
    assert.equal(asked, 0);
  });

  it("a Gemini key error asks the app for the key and keeps the server message", () => {
    let asked = 0;
    const message = enhanceFailureMessage(
      { code: "GEMINI_INVALID_API_KEY", message: "Key is invalid" },
      () => asked++,
    );
    assert.equal(message, "Key is invalid");
    assert.equal(asked, 1);
  });

  it("a Gemini key error with no app handler names where to add the key", () => {
    const message = enhanceFailureMessage({ code: "GEMINI_API_KEY_MISSING", message: "x" });
    assert.equal(message, MISSING_GEMINI_KEY_MESSAGE);
  });
});

describe("history after enhance", () => {
  it("keeps the typed prompt and adds the enhanced one as the current entry", () => {
    const next = historyAfterEnhance({
      history: [],
      index: -1,
      sourcePrompt: "a fox",
      enhancedPrompt: "a red fox at dawn",
    });
    assert.deepEqual(
      next.history.map((entry) => entry.text),
      ["a fox", "a red fox at dawn"],
    );
    assert.equal(next.history[next.index]?.text, "a red fox at dawn");
    assert.equal(next.history[next.index]?.provenance, "enhanced");
  });
});
