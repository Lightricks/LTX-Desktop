import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { ExploreAsset } from "../../../../lib/explore-contract.ts";
import { applyImportedAsset } from "./applyImportedAsset.ts";

function asset(id: string): ExploreAsset {
  return { id } as ExploreAsset;
}

function harness(expected = true) {
  const calls = { cached: [] as string[], accepted: [] as string[], rejected: 0 };
  return {
    calls,
    deps: {
      isExpectedAsset: () => expected,
      cacheAsset: (a: ExploreAsset) => calls.cached.push(a.id),
      accept: (a: ExploreAsset) => calls.accepted.push(a.id),
      reject: () => {
        calls.rejected += 1;
      },
    },
  };
}

describe("applyImportedAsset", () => {
  it("accepts directly without an import gate", async () => {
    const { calls, deps } = harness();
    await applyImportedAsset({ asset: asset("a"), ...deps });
    assert.deepEqual(calls.accepted, ["a"]);
  });

  it("rejects the wrong media kind without running the gate", async () => {
    const { calls, deps } = harness(false);
    let gated = false;
    await applyImportedAsset({
      asset: asset("a"),
      ...deps,
      prepareImport: async () => {
        gated = true;
      },
    });
    assert.equal(calls.rejected, 1);
    assert.equal(gated, false);
    assert.deepEqual(calls.accepted, []);
  });

  it("defers to the gate and accepts the asset it hands back", async () => {
    const { calls, deps } = harness();
    let finish: (() => void) | null = null;
    await applyImportedAsset({
      asset: asset("long"),
      ...deps,
      prepareImport: async (_source, accept) => {
        finish = () => accept(asset("trimmed"));
      },
    });
    assert.deepEqual(calls.accepted, [], "a pending trim keeps the old value");
    finish!();
    assert.deepEqual(calls.accepted, ["trimmed"]);
    assert.deepEqual(calls.cached, ["long", "trimmed"]);
  });
});
