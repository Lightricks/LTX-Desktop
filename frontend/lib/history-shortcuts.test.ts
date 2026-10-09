import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { historyShortcutAction, type HistoryShortcut } from "./history-shortcuts.ts";

const ready = {
  platform: "darwin",
  code: "BracketLeft",
  altKey: false,
  metaKey: true,
  ctrlKey: false,
  shiftKey: false,
  inTextEntry: false,
  modalOpen: false,
  canGoBack: true,
  canGoForward: true,
};

function shortcut(overrides: Partial<typeof ready>): HistoryShortcut | null {
  return historyShortcutAction({ ...ready, ...overrides });
}

describe("historyShortcutAction", () => {
  it("does not navigate while a modal is open", () => {
    assert.equal(shortcut({ modalOpen: true }), null);
  });

  it("does not go back from the start of the stack", () => {
    assert.equal(shortcut({ canGoBack: false }), null);
  });

  it("does not steal cmd-left from a text field on macOS", () => {
    assert.equal(shortcut({ code: "ArrowLeft", inTextEntry: true }), null);
  });

  it("treats ctrl-left as a word jump, not history", () => {
    assert.equal(
      shortcut({ platform: undefined, code: "ArrowLeft", metaKey: false, ctrlKey: true, altKey: false }),
      null,
    );
  });

  it("goes forward on macOS with cmd-bracket and cmd-right", () => {
    assert.equal(shortcut({ code: "BracketRight" }), "forward");
    assert.equal(shortcut({ code: "ArrowRight" }), "forward");
  });

  it("ignores alt-arrow on macOS", () => {
    assert.equal(shortcut({ code: "ArrowLeft", metaKey: false, altKey: true }), null);
    assert.equal(shortcut({ code: "ArrowRight", metaKey: false, altKey: true }), null);
  });

  it("does not navigate when shift is held", () => {
    assert.equal(shortcut({ shiftKey: true }), null);
    assert.equal(shortcut({ code: "BracketRight", shiftKey: true }), null);
  });

  it("uses alt-left and alt-right off macOS", () => {
    assert.equal(
      shortcut({ platform: undefined, code: "ArrowLeft", metaKey: false, altKey: true }),
      "back",
    );
    assert.equal(
      shortcut({
        platform: undefined,
        code: "ArrowRight",
        metaKey: false,
        altKey: true,
        canGoForward: false,
      }),
      null,
    );
  });
});
