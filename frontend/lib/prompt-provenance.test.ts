import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  type PromptHistoryEntry,
  appendEnhanceResult,
  historyForRestoredPrompt,
  parsePromptProvenance,
  resolvePromptProvenance,
  restorePromptHistory,
  shouldApplyEnhanceResult,
} from "./prompt-provenance.ts";

const typed = (text: string): PromptHistoryEntry => ({ text, provenance: "typed" });
const enhanced = (text: string): PromptHistoryEntry => ({ text, provenance: "enhanced" });

describe("resolvePromptProvenance", () => {
  it("reports typed for a prompt with no enhance history", () => {
    assert.equal(resolvePromptProvenance("a cat", [], -1), "typed");
  });

  it("reports enhanced while the prompt still matches the enhanced history entry", () => {
    const history = [typed("a cat"), enhanced("a long descriptive caption")];
    assert.equal(
      resolvePromptProvenance("a long descriptive caption", history, 1),
      "enhanced",
    );
  });

  it("reports typed once the user edits the enhanced text", () => {
    const history = [typed("a cat"), enhanced("a long descriptive caption")];
    assert.equal(
      resolvePromptProvenance("a long descriptive caption!", history, 1),
      "typed",
    );
  });

  it("reports typed after undo back to the originally typed entry", () => {
    const history = [typed("a cat"), enhanced("a long descriptive caption")];
    assert.equal(resolvePromptProvenance("a cat", history, 0), "typed");
  });

  it("reports typed for an out-of-range history index", () => {
    assert.equal(resolvePromptProvenance("a cat", [typed("a cat")], 5), "typed");
  });
});

describe("appendEnhanceResult", () => {
  it("records the source as typed and the result as enhanced", () => {
    const next = appendEnhanceResult({
      history: [],
      index: -1,
      sourcePrompt: "a cat",
      sourceProvenance: "typed",
      enhancedPrompt: "a long descriptive caption",
    });
    assert.deepEqual(next, {
      history: [typed("a cat"), enhanced("a long descriptive caption")],
      index: 1,
    });
    assert.equal(
      resolvePromptProvenance("a long descriptive caption", next.history, next.index),
      "enhanced",
    );
  });

  it("does not duplicate the source entry when it is already the current entry", () => {
    const next = appendEnhanceResult({
      history: [typed("a cat"), enhanced("first rewrite")],
      index: 1,
      sourcePrompt: "first rewrite",
      sourceProvenance: "enhanced",
      enhancedPrompt: "second rewrite",
    });
    assert.deepEqual(next, {
      history: [typed("a cat"), enhanced("first rewrite"), enhanced("second rewrite")],
      index: 2,
    });
  });

  it("drops the redo tail when enhancing after an undo", () => {
    const next = appendEnhanceResult({
      history: [typed("a cat"), enhanced("first rewrite")],
      index: 0,
      sourcePrompt: "a cat",
      sourceProvenance: "typed",
      enhancedPrompt: "fresh rewrite",
    });
    assert.deepEqual(next, {
      history: [typed("a cat"), enhanced("fresh rewrite")],
      index: 1,
    });
  });

  it("keeps an edited source prompt as typed", () => {
    const next = appendEnhanceResult({
      history: [typed("a cat"), enhanced("first rewrite")],
      index: 1,
      sourcePrompt: "first rewrite, edited",
      sourceProvenance: "typed",
      enhancedPrompt: "second rewrite",
    });
    assert.deepEqual(next.history[2], typed("first rewrite, edited"));
    assert.deepEqual(next.history[3], enhanced("second rewrite"));
    assert.equal(next.index, 3);
  });
});

describe("historyForRestoredPrompt", () => {
  it("reinstates enhanced provenance for a recovered in-flight generation", () => {
    const restored = historyForRestoredPrompt("a long descriptive caption", "enhanced");
    assert.deepEqual(restored, {
      history: [enhanced("a long descriptive caption")],
      index: 0,
    });
    assert.equal(
      resolvePromptProvenance(
        "a long descriptive caption",
        restored.history,
        restored.index,
      ),
      "enhanced",
    );
  });

  it("leaves the history empty for a typed prompt", () => {
    assert.deepEqual(historyForRestoredPrompt("a cat", "typed"), {
      history: [],
      index: -1,
    });
  });

  it("treats a marker written before provenance existed as typed", () => {
    assert.deepEqual(historyForRestoredPrompt("a cat", undefined), {
      history: [],
      index: -1,
    });
  });
});

describe("parsePromptProvenance", () => {
  it("accepts typed and enhanced", () => {
    assert.equal(parsePromptProvenance("typed"), "typed");
    assert.equal(parsePromptProvenance("enhanced"), "enhanced");
  });

  it("ignores other values", () => {
    assert.equal(parsePromptProvenance("other"), undefined);
    assert.equal(parsePromptProvenance(undefined), undefined);
  });
});

describe("shouldApplyEnhanceResult", () => {
  it("applies the rewrite only when the box still holds the source prompt", () => {
    assert.equal(shouldApplyEnhanceResult("a cat", "a cat"), true);
    assert.equal(shouldApplyEnhanceResult("a dog", "a cat"), false);
  });
});

describe("restorePromptHistory", () => {
  it("keeps in-session enhance history over the last generation", () => {
    const stored = {
      history: [typed("a cat"), enhanced("a long descriptive caption")],
      index: 1,
    };
    assert.deepEqual(
      restorePromptHistory({
        stored,
        prompt: "a long descriptive caption",
        lastGenerationPrompt: "something else",
        lastGenerationProvenance: "typed",
      }),
      stored,
    );
  });

  it("seeds enhanced provenance when the box still holds the last generation", () => {
    const restored = restorePromptHistory({
      stored: undefined,
      prompt: "a long descriptive caption",
      lastGenerationPrompt: "a long descriptive caption",
      lastGenerationProvenance: "enhanced",
    });
    assert.deepEqual(restored, {
      history: [enhanced("a long descriptive caption")],
      index: 0,
    });
    assert.equal(
      resolvePromptProvenance(
        "a long descriptive caption",
        restored.history,
        restored.index,
      ),
      "enhanced",
    );
  });

  it("does not treat a reset seed prompt as the last enhanced generation", () => {
    assert.deepEqual(
      restorePromptHistory({
        stored: { history: [], index: -1 },
        prompt: "mouse in a house",
        lastGenerationPrompt: "a rewritten felt palace",
        lastGenerationProvenance: "enhanced",
      }),
      { history: [], index: -1 },
    );
  });
});
