import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  assistantSelectionIssue,
  isAssistantSelectionReady,
} from "./assistant-selection.ts";

const models = [{ model: "qwen3.5:9b", installed: true }];
const notDownloaded = [{ model: "qwen3.5:9b", installed: false }];

describe("assistant selection send gate", () => {
  it("allows Gemini only when an API key is configured", () => {
    assert.equal(
      isAssistantSelectionReady({
        provider: "gemini",
        model: "qwen3.5:9b",
        hasGeminiApiKey: true,
        models,
      }),
      true,
    );
    assert.equal(
      assistantSelectionIssue({
        provider: "gemini",
        model: undefined,
        hasGeminiApiKey: false,
        models,
      }),
      "Add a Gemini API key in Settings to send messages.",
    );
  });

  it("allows a local model only when that checkpoint is installed", () => {
    assert.equal(
      isAssistantSelectionReady({
        provider: "local",
        model: "qwen3.5:9b",
        hasGeminiApiKey: false,
        models,
      }),
      true,
    );
    assert.equal(
      isAssistantSelectionReady({
        provider: "local",
        model: "qwen3.5:9b",
        hasGeminiApiKey: true,
        models: notDownloaded,
      }),
      false,
    );
    assert.equal(
      assistantSelectionIssue({
        provider: "local",
        model: "qwen3.5:9b",
        hasGeminiApiKey: true,
        models: notDownloaded,
      }),
      "Download this model in Settings, or select a valid model.",
    );
    assert.equal(
      assistantSelectionIssue({
        provider: "local",
        model: "qwen3.5:9b",
        hasGeminiApiKey: true,
        models: undefined,
      }),
      "Select a valid assistant model to send messages.",
    );
  });
});
