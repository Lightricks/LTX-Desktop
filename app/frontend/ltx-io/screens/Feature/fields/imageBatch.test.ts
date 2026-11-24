import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { ExploreAsset } from "../../../../lib/explore-contract.ts";
import { collectImageBatch, planImageBatch } from "./imageBatch.ts";

function asset(id: string): ExploreAsset {
  return { id } as ExploreAsset;
}

describe("planImageBatch", () => {
  it("keeps the first picks in order and counts the cut ones, not the whole batch", () => {
    assert.deepEqual(planImageBatch(["a", "b", "c", "d"], 3), {
      take: ["a", "b", "c"],
      overLimit: 1,
    });
  });

  it("cuts every pick when no slot is left, and never reports a negative count", () => {
    assert.deepEqual(planImageBatch(["a", "b"], 0), { take: [], overLimit: 2 });
    assert.deepEqual(planImageBatch(["a", "b"], -3), { take: [], overLimit: 2 });
  });
});

describe("collectImageBatch", () => {
  it("keeps the other images, in pick order, when one file fails or has the wrong kind", () => {
    const failure = new Error("could not read");
    const result = collectImageBatch(
      [
        { status: "fulfilled", value: asset("first") },
        { status: "rejected", reason: failure },
        { status: "fulfilled", value: asset("video") },
        { status: "fulfilled", value: asset("last") },
      ],
      (value) => value.id !== "video",
    );

    assert.deepEqual(
      result.assets.map((value) => value.id),
      ["first", "last"],
    );
    assert.equal(result.wrongKind, 1);
    assert.deepEqual(result.failures, [failure]);
  });
});
