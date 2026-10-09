import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  filtersFromSearchParams,
  lightboxAssetIdFromSearchParams,
  searchParamsFromFilters,
  searchParamsFromLibraryState,
  type AssetsListFilters,
} from "./assetsSearchParams.ts";

describe("assetsSearchParams", () => {
  it("omits default newest / all", () => {
    const params = searchParamsFromFilters({
      mediaKind: null,
      sort: "created_at-desc",
      q: "",
    });
    assert.equal(params.toString(), "");
  });

  it("round-trips kind, oldest, and q", () => {
    const filters: AssetsListFilters = {
      mediaKind: "audio",
      sort: "created_at-asc",
      q: "fox",
    };
    assert.deepEqual(filtersFromSearchParams(searchParamsFromFilters(filters)), filters);
  });

  it("ignores unknown params including t and origin", () => {
    const filters = filtersFromSearchParams(
      new URLSearchParams("media_kind=video&t=secret&sort=nope&origin=uploaded"),
    );
    assert.equal(filters.mediaKind, "video");
    assert.equal(filters.sort, "created_at-desc");
  });

  it("round-trips an open lightbox asset without dropping filters", () => {
    const filters: AssetsListFilters = {
      mediaKind: "video",
      sort: "created_at-desc",
      q: "",
    };
    const params = searchParamsFromLibraryState(filters, "asset-9");
    assert.equal(params.get("asset"), "asset-9");
    assert.equal(params.get("media_kind"), "video");
    assert.equal(lightboxAssetIdFromSearchParams(params), "asset-9");
    assert.equal(lightboxAssetIdFromSearchParams(searchParamsFromFilters(filters)), null);
    assert.equal(searchParamsFromLibraryState(filters, null).has("asset"), false);
  });
});
