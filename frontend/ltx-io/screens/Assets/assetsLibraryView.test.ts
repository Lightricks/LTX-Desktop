import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { resolveAssetsLibraryView } from "./assetsLibraryView.ts";

describe("resolveAssetsLibraryView", () => {
  it("uses the full error state when the first fetch fails with no items", () => {
    const view = resolveAssetsLibraryView({
      isLoading: false,
      error: new Error("boom"),
      loadedItemCount: 0,
      isFetchNextPageError: false,
    });

    assert.deepEqual(view, {
      kind: "error",
      inlineError: null,
      retryTarget: "refetch",
    });
  });

  it("keeps loaded items and retries the next page after a pagination failure", () => {
    const view = resolveAssetsLibraryView({
      isLoading: false,
      error: new Error("next page"),
      loadedItemCount: 8,
      isFetchNextPageError: true,
    });

    assert.deepEqual(view, {
      kind: "content",
      inlineError: "next-page",
      retryTarget: "fetchNextPage",
    });
  });

  it("keeps loaded items and retries the query after a refetch failure", () => {
    const view = resolveAssetsLibraryView({
      isLoading: false,
      error: new Error("refetch"),
      loadedItemCount: 8,
      isFetchNextPageError: false,
    });

    assert.deepEqual(view, {
      kind: "content",
      inlineError: "refetch",
      retryTarget: "refetch",
    });
  });

  it("does not treat a successful list as an error", () => {
    const view = resolveAssetsLibraryView({
      isLoading: false,
      error: null,
      loadedItemCount: 3,
      isFetchNextPageError: false,
    });

    assert.deepEqual(view, {
      kind: "content",
      inlineError: null,
      retryTarget: "refetch",
    });
  });

  it("keeps loading while an empty page still has a next cursor", () => {
    const view = resolveAssetsLibraryView({
      isLoading: false,
      error: null,
      loadedItemCount: 0,
      isFetchNextPageError: false,
      hasNextPage: true,
    });

    assert.deepEqual(view, {
      kind: "loading",
      inlineError: null,
      retryTarget: "refetch",
    });
  });
});
