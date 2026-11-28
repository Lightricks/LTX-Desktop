import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { DEEP_LINK_INTENT_TTL_MS, type DeepLinkIntent } from "../shared/deep-link.ts";

import { type DeepLinkInboxDeps, createDeepLinkInbox } from "./deep-link-inbox.ts";

function harness(argv: readonly string[] = []) {
  let now = 1_000;
  let nextId = 1;
  const published: DeepLinkIntent[] = [];
  let windows = 0;
  let openUrl: ((url: string) => void) | null = null;
  let secondInstance: ((commandLine: readonly string[]) => void) | null = null;
  const deps: DeepLinkInboxDeps = {
    argv,
    now: () => now,
    newId: () => String(nextId++),
    onOpenUrl: (listener) => {
      openUrl = listener;
    },
    onSecondInstance: (listener) => {
      secondInstance = listener;
    },
    ensureWindow: () => {
      windows += 1;
    },
    publish: (intent) => {
      published.push(intent);
    },
  };
  const inbox = createDeepLinkInbox(deps);
  return {
    inbox,
    published,
    get windows() {
      return windows;
    },
    advance: (ms: number) => {
      now += ms;
    },
    openUrl: (url: string) => {
      openUrl?.(url);
    },
    secondInstance: (commandLine: readonly string[]) => {
      secondInstance?.(commandLine);
    },
  };
}

describe("createDeepLinkInbox", () => {
  it("queues a cold-start argv link and ignores inspect flags", () => {
    const { inbox, published, windows } = harness([
      "Electron",
      ".",
      "--inspect=9229",
      "ltxdesktop://fetcher/t2v",
    ]);
    const pending = inbox.take();
    assert.equal(pending?.target.kind, "fetcher-tool");
    assert.equal(pending?.target.slug, "t2v");
    assert.equal(pending?.arrival, "cold-start");
    assert.equal(published.length, 1);
    assert.equal(windows, 1);
    assert.equal(inbox.take(), null);
  });

  it("does not clear a queued intent when later argv has no link", () => {
    const { inbox, secondInstance } = harness(["ltxdesktop://fetcher/t2v"]);
    secondInstance(["Electron", "."]);
    assert.equal(inbox.take()?.target.slug, "t2v");
  });

  it("latest-wins: take returns the newest intent and clears the inbox", () => {
    const { inbox, openUrl } = harness();
    openUrl("ltxdesktop://fetcher/t2v");
    openUrl("ltxdesktop://fetcher/i2v");
    assert.equal(inbox.take()?.target.slug, "i2v");
    assert.equal(inbox.take(), null);
  });

  it("expires unclaimed intents at take time", () => {
    const { inbox, advance } = harness(["ltxdesktop://fetcher/t2v"]);
    advance(DEEP_LINK_INTENT_TTL_MS);
    assert.equal(inbox.take(), null);
  });

  it("second-instance queues a link and focuses exactly once", () => {
    const session = harness();
    session.secondInstance(["LTX Desktop.exe", "ltxdesktop://fetcher/i2v"]);
    assert.equal(session.inbox.take()?.target.slug, "i2v");
    assert.equal(session.windows, 1);
  });

  it("focuses an existing window when the second instance has no URL", () => {
    const session = harness();
    session.secondInstance(["LTX Desktop.exe"]);
    assert.equal(session.inbox.take(), null);
    assert.equal(session.windows, 1);
  });

  it("queues Assets as a Home-shell destination", () => {
    const { inbox, published } = harness(["ltxdesktop://assets"]);
    const pending = inbox.take();
    assert.deepEqual(pending?.target, { kind: "home-shell", slug: "assets" });
    assert.equal(published.length, 1);
  });
});
