import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  addKeyframes,
  freeFrameFor,
  keyframePatchForKey,
  markerSizeForCount,
  orderLightboxAssets,
  spreadKeyframes,
  stillIdAtPosition,
  stillPosition,
  swapKeyframeFrames,
} from "./keyframesModel.ts";

describe("swapKeyframeFrames", () => {
  const still = (id: string, frameIndex: number, strength = 1) => ({
    id,
    asset: { assetId: `asset-${id}` },
    frameIndex,
    strength,
  });

  it("gives each still the frame of the other and keeps its image and strength", () => {
    const next = swapKeyframeFrames(
      [still("a", 0, 1), still("b", 36, 0.5), still("c", 72, 0.8)],
      "a",
      "c",
    );

    assert.deepEqual(
      next.map(({ id, frameIndex, strength, asset }) => [id, frameIndex, strength, asset.assetId]),
      [
        ["a", 72, 1, "asset-a"],
        ["b", 36, 0.5, "asset-b"],
        ["c", 0, 0.8, "asset-c"],
      ],
    );
  });

  it("never puts two stills on one frame", () => {
    const next = swapKeyframeFrames([still("a", 5), still("b", 9), still("c", 20)], "b", "a");

    assert.equal(new Set(next.map((item) => item.frameIndex)).size, 3);
  });

  it("changes nothing for an unknown id or the same id twice", () => {
    const list = [still("a", 0), still("b", 36)];

    assert.deepEqual(swapKeyframeFrames(list, "a", "nope"), list);
    assert.deepEqual(swapKeyframeFrames(list, "a", "a"), list);
  });

  it("does not change the list it was given", () => {
    const list = [still("a", 0), still("b", 36)];
    swapKeyframeFrames(list, "a", "b");

    assert.deepEqual(list.map((item) => item.frameIndex), [0, 36]);
  });
});

describe("markerSizeForCount", () => {
  it("gives the biggest marker for 5 stills or fewer, a medium one for 6 to 8, a small one for 9 or more", () => {
    assert.deepEqual(
      [0, 1, 5, 6, 8, 9, 10].map(markerSizeForCount),
      [44, 44, 44, 36, 36, 28, 28],
    );
  });

  it("never makes a marker bigger as stills are added", () => {
    const sizes = Array.from({ length: 12 }, (_, count) => markerSizeForCount(count));

    assert.deepEqual(sizes, [...sizes].sort((a, b) => b - a));
  });
});

function ids() {
  let next = 0;
  return () => `id-${next++}`;
}

const assets = (count: number) =>
  Array.from({ length: count }, (_, index) => ({ assetId: `asset-${index}` }));

describe("addKeyframes", () => {
  it("never puts two stills on one frame when a batch asks for the same one", () => {
    const next = addKeyframes([], assets(3), {
      lastFrame: 96,
      preferredFrame: 0,
      placement: "spread",
      maxCount: 10,
      createId: ids(),
    });

    const frames = next.map((item) => item.frameIndex);
    assert.equal(new Set(frames).size, 3);
    assert.equal(frames[0], 0);
  });

  it("stops at the most the field takes", () => {
    const next = addKeyframes([], assets(5), {
      lastFrame: 96,
      preferredFrame: 0,
      placement: "spread",
      maxCount: 2,
      createId: ids(),
    });

    assert.equal(next.length, 2);
  });

  it("keeps a still dropped on the timeline next to the drop, not in the widest gap", () => {
    const existing = [
      { id: "a", asset: { assetId: "a" }, frameIndex: 50, strength: 1 },
    ];
    const next = addKeyframes(existing, assets(1), {
      lastFrame: 96,
      preferredFrame: 50,
      placement: "near",
      maxCount: 10,
      createId: ids(),
    });

    assert.ok(Math.abs(next[1].frameIndex - 50) <= 1);
  });

  it("keeps every dropped still when the drop frame is taken, and may place it to the right", () => {
    const existing = [
      { id: "a", asset: { assetId: "a" }, frameIndex: 0, strength: 1 },
    ];
    const next = addKeyframes(existing, assets(3), {
      lastFrame: 144,
      preferredFrame: 0,
      placement: "near",
      maxCount: 10,
      createId: ids(),
    });

    assert.deepEqual(
      next.map((item) => item.frameIndex),
      [0, 1, 2, 3],
    );
  });

  it("starts the search at the drop frame, not at the last frame", () => {
    const next = addKeyframes([], assets(1), {
      lastFrame: 144,
      preferredFrame: 100,
      placement: "near",
      maxCount: 10,
      createId: ids(),
    });

    assert.equal(next[0].frameIndex, 100);
  });
});

describe("spreadKeyframes", () => {
  const still = (id: string, frameIndex: number) => ({
    id,
    asset: { assetId: id },
    frameIndex,
    strength: 1,
  });

  it("puts the stills an equal gap apart from the first frame to the last, keeping their order", () => {
    // Listed out of time order on purpose: the list keeps the order the user added them in.
    const next = spreadKeyframes([still("c", 90), still("a", 3), still("b", 10)], 96);

    assert.deepEqual(
      next.map((item) => [item.id, item.frameIndex]),
      [
        ["c", 96],
        ["a", 0],
        ["b", 48],
      ],
    );
  });

  it("changes nothing for one still, or when the clip has too few frames for them", () => {
    const one = [still("a", 12)];
    assert.deepEqual(spreadKeyframes(one, 96), one);
    const three = [still("a", 0), still("b", 1), still("c", 2)];
    assert.deepEqual(spreadKeyframes(three, 1), three);
  });
});

describe("orderLightboxAssets", () => {
  it("gives each still its own place when one image sits at two times", () => {
    const loaded = new Map([
      ["a", "asset-a"],
      ["b", "asset-b"],
    ]);
    assert.deepEqual(orderLightboxAssets(["a", "b", "a"], loaded), {
      items: ["asset-a", "asset-b", "asset-a"],
      positions: [0, 1, 2],
    });
  });

  it("keeps the still order, and marks a still whose asset has not loaded", () => {
    const loaded = new Map([
      ["c", "asset-c"],
      ["a", "asset-a"],
    ]);
    assert.deepEqual(orderLightboxAssets(["c", "b", "a"], loaded), {
      items: ["asset-c", "asset-a"],
      positions: [0, null, 1],
    });
  });
});

describe("freeFrameFor and keyframePatchForKey", () => {
  const still = (id: string, frameIndex: number, strength = 0.5) => ({
    id,
    asset: { assetId: `asset-${id}` },
    frameIndex,
    strength,
  });
  const list = [still("a", 10), still("b", 11)];

  it("does not count the frame of the still itself as taken", () => {
    assert.equal(freeFrameFor(list, "a", 10, 144), 10);
    assert.equal(freeFrameFor(list, "a", 11, 144), 12);
  });

  it("moves a still by one frame with the left and right keys, over a taken frame", () => {
    assert.deepEqual(keyframePatchForKey(list, list[0], "ArrowLeft", 144), { frameIndex: 9 });
    assert.deepEqual(keyframePatchForKey(list, list[0], "ArrowRight", 144), { frameIndex: 12 });
  });

  it("changes the strength by one step with the up and down keys, and stays in range", () => {
    assert.deepEqual(keyframePatchForKey(list, list[0], "ArrowUp", 144), { strength: 0.55 });
    assert.deepEqual(keyframePatchForKey(list, list[0], "ArrowDown", 144), { strength: 0.45 });
    const full = still("c", 30, 1);
    assert.deepEqual(keyframePatchForKey([full], full, "ArrowUp", 144), { strength: 1 });
  });

  it("changes nothing for any other key", () => {
    assert.equal(keyframePatchForKey(list, list[0], "Enter", 144), null);
  });
});

describe("stillPosition and stillIdAtPosition", () => {
  // The second still has no loaded asset, so it has no place in the preview.
  const ids = ["a", "b", "c"];
  const positions = [0, null, 1];

  it("finds the place of a still, and none for a still with no image or a still not listed", () => {
    assert.equal(stillPosition(ids, positions, "c"), 1);
    assert.equal(stillPosition(ids, positions, "b"), null);
    assert.equal(stillPosition(ids, positions, "gone"), null);
  });

  it("finds the still at a place, the reverse of the above", () => {
    assert.equal(stillIdAtPosition(ids, positions, 1), "c");
    assert.equal(stillIdAtPosition(ids, positions, 5), undefined);
  });
});
