import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { resolveUsableEnhanceProvider } from "./enhance-provider.ts";

describe("resolveUsableEnhanceProvider", () => {
  it("falls back from stored api to local when the Gemini key is gone", () => {
    assert.equal(
      resolveUsableEnhanceProvider({
        preference: "api",
        hasGeminiApiKey: false,
        hasLocalTextEncoder: true,
        fallback: "api",
      }),
      "local",
    );
  });

  it("falls back from stored local to api when Gemma is unavailable", () => {
    assert.equal(
      resolveUsableEnhanceProvider({
        preference: "local",
        hasGeminiApiKey: true,
        hasLocalTextEncoder: false,
        fallback: "local",
      }),
      "api",
    );
  });

  it("keeps the status default when nothing is stored", () => {
    assert.equal(
      resolveUsableEnhanceProvider({
        preference: null,
        hasGeminiApiKey: true,
        hasLocalTextEncoder: true,
        fallback: "api",
      }),
      "api",
    );
  });
});
