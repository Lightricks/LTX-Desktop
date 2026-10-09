import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  findLightboxAssetIndex,
  lightboxMediaKey,
  nextLightboxIndex,
  resolveLightboxDeepLinkAction,
  shouldOpenAssetFromKey,
  shouldHandleLightboxArrowKey,
} from "./lightboxCollection.ts";

describe("nextLightboxIndex", () => {
  it("advances within the loaded collection", () => {
    assert.equal(nextLightboxIndex(0, 2, false, false), "next");
  });

  it("fetches another page at the end of loaded items", () => {
    assert.equal(nextLightboxIndex(1, 2, true, false), "fetch");
  });

  it("does not start another boundary fetch while one is running", () => {
    assert.equal(nextLightboxIndex(1, 2, true, true), "end");
  });

  it("ends at the final available item", () => {
    assert.equal(nextLightboxIndex(1, 2, false, false), "end");
  });
});

describe("findLightboxAssetIndex", () => {
  const items = [{ id: "first" }, { id: "second" }];

  it("resolves the active asset by ID after the collection changes", () => {
    assert.equal(findLightboxAssetIndex([items[1]!, items[0]!], "second"), 0);
  });

  it("reports a removed active asset", () => {
    assert.equal(findLightboxAssetIndex([items[0]!], "second"), -1);
  });
});

describe("shouldHandleLightboxArrowKey", () => {
  it("handles a navigable arrow key outside interactive media and controls", () => {
    assert.equal(
      shouldHandleLightboxArrowKey("ArrowRight", "content", true),
      true,
    );
  });

  it("allows navigation from lightbox chrome controls", () => {
    assert.equal(
      shouldHandleLightboxArrowKey("ArrowRight", "chrome", true),
      true,
    );
  });

  it("does not hijack media, form controls, or unavailable navigation", () => {
    assert.equal(
      shouldHandleLightboxArrowKey("ArrowRight", "blocked", true),
      false,
    );
    assert.equal(
      shouldHandleLightboxArrowKey("ArrowLeft", "content", false),
      false,
    );
    assert.equal(
      shouldHandleLightboxArrowKey("Enter", "content", true),
      false,
    );
  });
});

describe("shouldOpenAssetFromKey", () => {
  it("opens its own role-button tile for Enter and Space", () => {
    assert.equal(shouldOpenAssetFromKey("Enter", "tile"), true);
    assert.equal(shouldOpenAssetFromKey(" ", "tile"), true);
  });

  it("does not open the tile from a nested control", () => {
    assert.equal(shouldOpenAssetFromKey("Enter", "nested"), false);
  });
});

describe("lightboxMediaKey", () => {
  it("remounts media when the active asset or source changes", () => {
    assert.notEqual(
      lightboxMediaKey("first", "/media/first.png"),
      lightboxMediaKey("second", "/media/second.png"),
    );
    assert.notEqual(
      lightboxMediaKey("first", "/media/first.png"),
      lightboxMediaKey("first", "/media/first-updated.png"),
    );
  });
});

describe("resolveLightboxDeepLinkAction", () => {
  const unresolved = {
    lightboxAssetId: "missing",
    lightboxIndex: -1,
    isLoading: false,
    isFetchingNextPage: false,
    isFetchNextPageError: false,
    hasNextPage: true,
  } as const;

  it("pages until the deep-linked asset is in the loaded collection", () => {
    assert.equal(resolveLightboxDeepLinkAction(unresolved), "fetchNextPage");
  });

  it("does not retry automatically after a next-page error", () => {
    assert.equal(
      resolveLightboxDeepLinkAction({
        ...unresolved,
        isFetchNextPageError: true,
      }),
      "idle",
    );
  });

  it("waits while a page is already loading", () => {
    assert.equal(
      resolveLightboxDeepLinkAction({
        ...unresolved,
        isFetchingNextPage: true,
      }),
      "idle",
    );
  });

  it("closes when the id is absent from every loaded page", () => {
    assert.equal(
      resolveLightboxDeepLinkAction({
        ...unresolved,
        hasNextPage: false,
      }),
      "close",
    );
  });

  it("stays idle once the asset is in the collection", () => {
    assert.equal(
      resolveLightboxDeepLinkAction({
        ...unresolved,
        lightboxIndex: 3,
        hasNextPage: false,
      }),
      "idle",
    );
  });
});
