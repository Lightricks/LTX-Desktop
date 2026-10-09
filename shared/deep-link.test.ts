import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  DEEP_LINK_INTENT_TTL_MS,
  DEEP_LINK_SHELL_SLUGS,
  type DeepLinkIntent,
  FETCHER_TOOL_SLUGS,
  findDeepLinkInArgv,
  isDeepLinkIntentFresh,
  isDeepLinkShellSlug,
  isFetcherToolId,
  parseDeepLink,
} from "./deep-link.ts";

describe("parseDeepLink", () => {
  it("parses ltxdesktop://fetcher/{tool} for shipped tools", () => {
    assert.deepEqual(parseDeepLink("ltxdesktop://fetcher/t2v"), {
      kind: "fetcher-tool",
      slug: "t2v",
    });
    assert.deepEqual(parseDeepLink("ltxdesktop://fetcher/i2v"), {
      kind: "fetcher-tool",
      slug: "i2v",
    });
    assert.deepEqual(parseDeepLink("ltxdesktop://fetcher/a2v"), {
      kind: "fetcher-tool",
      slug: "a2v",
    });
    assert.deepEqual(parseDeepLink("ltxdesktop://fetcher/retake"), {
      kind: "fetcher-tool",
      slug: "retake",
    });
    assert.deepEqual(parseDeepLink("ltxdesktop://fetcher/extend"), {
      kind: "fetcher-tool",
      slug: "extend",
    });
    assert.deepEqual(parseDeepLink("ltxdesktop://fetcher/cozy-felt"), {
      kind: "fetcher-tool",
      slug: "cozy-felt",
    });
    assert.deepEqual(parseDeepLink("ltxdesktop://fetcher/dolly-in"), {
      kind: "fetcher-tool",
      slug: "dolly-in",
    });
  });

  it("parses ltxdesktop://assets as a Home-shell destination", () => {
    assert.deepEqual(parseDeepLink("ltxdesktop://assets"), {
      kind: "home-shell",
      slug: "assets",
    });
    assert.deepEqual(parseDeepLink("ltxdesktop:///assets"), {
      kind: "home-shell",
      slug: "assets",
    });
    assert.deepEqual(parseDeepLink("ltxdesktop://ASSETS"), {
      kind: "home-shell",
      slug: "assets",
    });
  });

  it("accepts an empty host form used by some senders", () => {
    assert.deepEqual(parseDeepLink("ltxdesktop:///fetcher/i2v"), {
      kind: "fetcher-tool",
      slug: "i2v",
    });
  });

  it("lowercases the slug", () => {
    assert.deepEqual(parseDeepLink("ltxdesktop://fetcher/T2V"), {
      kind: "fetcher-tool",
      slug: "t2v",
    });
  });

  it("rejects other schemes, missing tools, extra segments, and unshipped slugs", () => {
    assert.equal(parseDeepLink("https://fetcher/t2v"), null);
    assert.equal(parseDeepLink("ltxdesktop://home/t2v"), null);
    assert.equal(parseDeepLink("ltxdesktop://fetcher/assets"), null);
    assert.equal(parseDeepLink("ltxdesktop://feature/assets"), null);
    assert.equal(parseDeepLink("ltxdesktop://assets/extra"), null);
    assert.equal(parseDeepLink("ltxdesktop://fetcher"), null);
    assert.equal(parseDeepLink("ltxdesktop://fetcher/t2v/extra"), null);
    assert.equal(parseDeepLink("ltxdesktop://projects/abc"), null);
    assert.equal(parseDeepLink("not a url"), null);
    assert.equal(parseDeepLink("ltxdesktop://fetcher/ic-lora"), null);
    assert.equal(parseDeepLink("ltxdesktop://fetcher/cozy"), null);
    assert.equal(parseDeepLink("ltxdesktop://fetcher/T2V_TOOL"), null);
    assert.equal(parseDeepLink("ltxdesktop://fetcher/"), null);
  });

  it("strips wrapping quotes used on Windows argv", () => {
    assert.deepEqual(parseDeepLink('"ltxdesktop://fetcher/t2v"'), {
      kind: "fetcher-tool",
      slug: "t2v",
    });
    assert.deepEqual(parseDeepLink("'ltxdesktop://fetcher/i2v'"), {
      kind: "fetcher-tool",
      slug: "i2v",
    });
  });
});

describe("isFetcherToolId", () => {
  it("accepts only the shipped fetcher tools", () => {
    assert.equal(isFetcherToolId("t2v"), true);
    assert.equal(isFetcherToolId("i2v"), true);
    assert.equal(isFetcherToolId("a2v"), true);
    assert.equal(isFetcherToolId("retake"), true);
    assert.equal(isFetcherToolId("extend"), true);
    assert.equal(isFetcherToolId("cozy-felt"), true);
    assert.equal(isFetcherToolId("dolly-in"), true);
    assert.equal(isFetcherToolId("assets"), false);
    assert.equal(isFetcherToolId("cozy"), false);
    assert.equal(isFetcherToolId("T2V"), false);
    assert.equal(isFetcherToolId(undefined), false);
  });
});

describe("isDeepLinkShellSlug", () => {
  it("accepts Assets and not generation tools", () => {
    assert.equal(isDeepLinkShellSlug("assets"), true);
    assert.equal(isDeepLinkShellSlug("t2v"), false);
    assert.equal(isDeepLinkShellSlug("projects"), false);
    assert.equal(isDeepLinkShellSlug(undefined), false);
    assert.equal(
      FETCHER_TOOL_SLUGS.some((slug) =>
        (DEEP_LINK_SHELL_SLUGS as readonly string[]).includes(slug),
      ),
      false,
    );
  });
});

describe("findDeepLinkInArgv", () => {
  it("returns the first valid deep link and ignores noise", () => {
    assert.deepEqual(
      findDeepLinkInArgv([
        "/path/to/Electron",
        ".",
        "--inspect=9229",
        "ltxdesktop://fetcher/t2v",
        "ltxdesktop://fetcher/i2v",
      ]),
      { kind: "fetcher-tool", slug: "t2v" },
    );
    assert.equal(findDeepLinkInArgv(["/path/to/Electron", "."]), null);
    assert.deepEqual(
      findDeepLinkInArgv(["LTX Desktop.exe", '"ltxdesktop://fetcher/i2v"']),
      { kind: "fetcher-tool", slug: "i2v" },
    );
  });
});

describe("isDeepLinkIntentFresh", () => {
  it("drops intents at the TTL", () => {
    const target = parseDeepLink("ltxdesktop://fetcher/t2v");
    assert.ok(target);
    const intent: DeepLinkIntent = {
      id: "1",
      target,
      receivedAt: 1_000,
      arrival: "cold-start",
    };
    assert.equal(isDeepLinkIntentFresh(intent, 1_000), true);
    assert.equal(
      isDeepLinkIntentFresh(intent, 1_000 + DEEP_LINK_INTENT_TTL_MS - 1),
      true,
    );
    assert.equal(isDeepLinkIntentFresh(intent, 1_000 + DEEP_LINK_INTENT_TTL_MS), false);
  });
});
