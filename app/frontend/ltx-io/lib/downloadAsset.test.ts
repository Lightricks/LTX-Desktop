import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { downloadAsset } from "./downloadAsset.ts";

type Asset = { name: string; url: string | null };

function run(overrides: { fetchAsset?: () => Promise<Asset>; fresh?: Asset } = {}) {
  const events: string[] = [];
  const fresh = overrides.fresh ?? { name: "clip.mp4", url: "/bytes?sig=new" };
  return {
    events,
    done: downloadAsset({
      fetchAsset: overrides.fetchAsset ?? (async () => fresh),
      mediaUrlForAsset: (asset) => asset.url,
      save: (url, filename) => events.push(`save ${url} ${filename}`),
      onError: () => events.push("error"),
    }),
  };
}

describe("downloadAsset", () => {
  it("saves the file from the URL of the asset it fetched again", async () => {
    const { events, done } = run();
    await done;
    assert.deepEqual(events, ["save /bytes?sig=new clip.mp4"]);
  });

  it("reports an error and saves nothing when the fetch fails", async () => {
    const { events, done } = run({
      fetchAsset: async () => {
        throw new Error("offline");
      },
    });
    await done;
    assert.deepEqual(events, ["error"]);
  });

  it("reports an error when the fresh asset has no URL", async () => {
    const { events, done } = run({ fresh: { name: "clip.mp4", url: null } });
    await done;
    assert.deepEqual(events, ["error"]);
  });
});
