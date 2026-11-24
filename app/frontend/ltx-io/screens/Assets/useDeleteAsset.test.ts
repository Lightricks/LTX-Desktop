import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { QueryClient } from "@tanstack/react-query";

import type { ExploreAssetListResponse } from "@/lib/explore-contract";

import { assetQueryKeys } from "../../hooks/assetQueryKeys.ts";
import {
  assetListQueryFilter,
  optimisticallyRemoveAssetFromListCache,
  removeAssetFromInfinitePages,
  restoreAssetCache,
} from "./assetDeleteCache.ts";

describe("assetListQueryFilter", () => {
  it("prefix-matches every asset list query key", () => {
    const listKey = assetQueryKeys.list({ sort: "created_at-desc", q: "" });
    assert.deepEqual(
      listKey.slice(0, assetListQueryFilter.queryKey.length),
      [...assetListQueryFilter.queryKey],
    );
  });
});

describe("removeAssetFromInfinitePages", () => {
  it("removes the asset from every cached infinite-query page without mutating the original", () => {
    const initial = {
      pages: [
        {
          items: [
            { id: "asset-1", in_use: false },
            { id: "asset-2", in_use: false },
          ],
          next_cursor: "cursor-2",
        },
        {
          items: [
            { id: "asset-2", in_use: false },
            { id: "asset-3", in_use: false },
          ],
          next_cursor: null,
        },
      ] satisfies ExploreAssetListResponse[],
      pageParams: [null, "cursor-2"],
    };

    const result = removeAssetFromInfinitePages(initial, "asset-2");

    assert.deepEqual(
      result.pages.map((page) => page.items.map((item) => item.id)),
      [["asset-1"], ["asset-3"]],
    );
    assert.deepEqual(
      initial.pages.map((page) => page.items.map((item) => item.id)),
      [
        ["asset-1", "asset-2"],
        ["asset-2", "asset-3"],
      ],
    );
    assert.deepEqual(result.pageParams, [null, "cursor-2"]);
  });

  it("updates list pages without touching cached asset details", async () => {
    const queryClient = new QueryClient();
    const listKey = assetQueryKeys.list({ sort: "created_at-desc", q: "" });
    const detailKey = assetQueryKeys.detail("asset-1");
    const listData = {
      pages: [{ items: [{ id: "asset-1", in_use: false }] }],
      pageParams: [null],
    };
    const detailData = {
      id: "asset-1",
      name: "asset-1.mp4",
      path: "/assets/asset-1.mp4",
    };

    queryClient.setQueryData(listKey, listData);
    queryClient.setQueryData(detailKey, detailData);

    const snapshots = await optimisticallyRemoveAssetFromListCache(
      queryClient,
      "asset-1",
    );

    assert.deepEqual(queryClient.getQueryData(listKey), {
      pages: [{ items: [] }],
      pageParams: [null],
    });
    assert.deepEqual(queryClient.getQueryData(detailKey), detailData);
    assert.deepEqual(snapshots, [[listKey, listData]]);
  });

  it("restores every cached list after a failed delete", async () => {
    const queryClient = new QueryClient();
    const firstKey = assetQueryKeys.list({ sort: "created_at-desc", q: "" });
    const secondKey = assetQueryKeys.list({ sort: "created_at-asc", q: "" });
    const firstPage = { items: [{ id: "asset-1", in_use: false }] };
    const secondPage = { items: [{ id: "asset-2", in_use: false }] };

    queryClient.setQueryData(firstKey, {
      pages: [firstPage],
      pageParams: [null],
    });
    queryClient.setQueryData(secondKey, {
      pages: [secondPage],
      pageParams: [null],
    });
    const snapshots = await optimisticallyRemoveAssetFromListCache(
      queryClient,
      "asset-1",
    );

    restoreAssetCache(queryClient, snapshots);

    assert.deepEqual(queryClient.getQueryData(firstKey), {
      pages: [firstPage],
      pageParams: [null],
    });
    assert.deepEqual(queryClient.getQueryData(secondKey), {
      pages: [secondPage],
      pageParams: [null],
    });
  });
});
