import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  assetQueryKeys,
  canonicalAssetSearchQuery,
  isImageAsset,
  isVideoAsset,
} from "./assetQueryKeys.ts";

describe("asset query keys", () => {
  it("canonicalizes search text before creating the list cache key", () => {
    assert.deepEqual(
      assetQueryKeys.list({ sort: "created_at-desc", q: "  fox  " }),
      ["assets", "list", { sort: "created_at-desc", q: "fox" }],
    );
  });

  it("omits whitespace-only search text from the default cache key", () => {
    assert.equal(canonicalAssetSearchQuery("   "), undefined);
    assert.deepEqual(
      assetQueryKeys.list({ sort: "created_at-desc", q: "   " }),
      ["assets", "list", { sort: "created_at-desc" }],
    );
  });
});

describe("asset query key utilities", () => {
  it("scopes a lookup key to a durable asset id", () => {
    assert.deepEqual(assetQueryKeys.detail("asset-22"), [
      "assets",
      "detail",
      "asset-22",
    ]);
  });

  it("accepts ingested images and rejects other media kinds", () => {
    assert.equal(isImageAsset({ media_kind: "image" }), true);
    assert.equal(isImageAsset({ media_kind: "video" }), false);
    assert.equal(isImageAsset({ media_kind: "audio" }), false);
  });

  it("accepts ingested videos and rejects other media kinds", () => {
    assert.equal(isVideoAsset({ media_kind: "video" }), true);
    assert.equal(isVideoAsset({ media_kind: "image" }), false);
    assert.equal(isVideoAsset({ media_kind: "audio" }), false);
  });
});
