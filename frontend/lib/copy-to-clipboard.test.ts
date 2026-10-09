import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";

import { copyTextToClipboard } from "./copy-to-clipboard.ts";

type Globals = { navigator?: unknown; document?: unknown };
const globals = globalThis as unknown as Globals;
const original = { navigator: globals.navigator, document: globals.document };

function setNavigator(value: unknown) {
  Object.defineProperty(globalThis, "navigator", { value, configurable: true });
}

function fakeDocument(execResult: boolean | Error) {
  const appended: unknown[] = [];
  const focused: string[] = [];
  const activeElement = { focus: () => focused.push("restored") };
  const textarea = {
    value: "",
    style: {} as Record<string, string>,
    setAttribute() {},
    select() {},
    setSelectionRange() {},
  };
  return {
    appended,
    focused,
    textarea,
    document: {
      activeElement,
      createElement: () => textarea,
      body: {
        appendChild: (node: unknown) => appended.push(node),
        removeChild: () => {},
      },
      execCommand: () => {
        if (execResult instanceof Error) throw execResult;
        return execResult;
      },
    },
  };
}

describe("copyTextToClipboard", () => {
  afterEach(() => {
    setNavigator(original.navigator);
    Object.defineProperty(globalThis, "document", {
      value: original.document,
      configurable: true,
    });
  });

  it("uses the async Clipboard API when it is available", async () => {
    const written: string[] = [];
    setNavigator({ clipboard: { writeText: async (v: string) => void written.push(v) } });

    assert.equal(await copyTextToClipboard("a prompt"), true);
    assert.deepEqual(written, ["a prompt"]);
  });

  it("falls back to execCommand when the API is missing (plain http on the LAN)", async () => {
    setNavigator({});
    const fake = fakeDocument(true);
    Object.defineProperty(globalThis, "document", { value: fake.document, configurable: true });

    assert.equal(await copyTextToClipboard("a prompt"), true);
    assert.equal(fake.textarea.value, "a prompt");
    assert.equal(fake.appended.length, 1);
    assert.deepEqual(fake.focused, ["restored"]);
  });

  it("reports failure when both paths fail", async () => {
    setNavigator({ clipboard: { writeText: async () => { throw new Error("denied"); } } });
    const fake = fakeDocument(false);
    Object.defineProperty(globalThis, "document", { value: fake.document, configurable: true });

    assert.equal(await copyTextToClipboard("a prompt"), false);

    const throwing = fakeDocument(new Error("blocked"));
    Object.defineProperty(globalThis, "document", { value: throwing.document, configurable: true });
    assert.equal(await copyTextToClipboard("a prompt"), false);
  });
});
