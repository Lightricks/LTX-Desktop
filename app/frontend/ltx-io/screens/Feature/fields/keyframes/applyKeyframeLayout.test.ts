import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { applyKeyframeLayout } from "./keyframesModel.ts";

describe("applyKeyframeLayout", () => {
  const still = (id: string, frameIndex: number, strength = 1) => ({
    id,
    asset: { assetId: `asset-${id}` },
    frameIndex,
    strength,
  });
  const options = { lastFrame: 144, maxCount: 10 };
  const frames = (list: readonly { id: string; frameIndex: number }[]) =>
    list.map((item) => [item.id, item.frameIndex]);

  it("takes frames and strengths from the draft and keeps images and list order", () => {
    const base = [still("a", 0), still("b", 36), still("c", 72)];
    const draft = [still("c", 0, 0.4), still("a", 72), still("b", 36)];

    const next = applyKeyframeLayout(base, base, draft, options);

    assert.deepEqual(
      next.map(({ id, frameIndex, strength, asset }) => [id, frameIndex, strength, asset.assetId]),
      [
        ["a", 72, 1, "asset-a"],
        ["b", 36, 1, "asset-b"],
        ["c", 0, 0.4, "asset-c"],
      ],
    );
  });

  it("removes a still that the user removed from the draft", () => {
    const base = [still("a", 0), still("b", 36), still("c", 72)];

    const next = applyKeyframeLayout(base, base, [still("a", 0), still("c", 72)], options);

    assert.deepEqual(frames(next), [["a", 0], ["c", 72]]);
  });

  it("adds a still that the user added to the draft, with its own id and image", () => {
    const base = [still("a", 0)];

    const next = applyKeyframeLayout(base, base, [still("a", 0), still("new", 40, 0.6)], options);

    assert.deepEqual(next.map(({ id, frameIndex, strength }) => [id, frameIndex, strength]), [
      ["a", 0, 1],
      ["new", 40, 0.6],
    ]);
  });

  it("keeps a still that was added to the list meanwhile", () => {
    const base = [still("a", 0)];
    const current = [still("a", 0), still("meanwhile", 50)];

    const next = applyKeyframeLayout(current, base, [still("a", 10)], options);

    assert.deepEqual(frames(next), [["a", 10], ["meanwhile", 50]]);
  });

  it("moves a draft still off the frame that a still added meanwhile took", () => {
    const base = [still("a", 0)];
    const current = [still("a", 0), still("meanwhile", 50)];

    const moved = applyKeyframeLayout(current, base, [still("a", 50)], options);
    const added = applyKeyframeLayout(current, base, [still("a", 0), still("new", 50)], options);

    assert.deepEqual(frames(moved), [["a", 51], ["meanwhile", 50]]);
    assert.equal(new Set(added.map((item) => item.frameIndex)).size, 3);
  });

  it("drops a still added past the maximum", () => {
    const base = [still("a", 0)];
    const draft = [still("a", 0), still("n1", 10), still("n2", 20)];

    const next = applyKeyframeLayout(base, base, draft, { lastFrame: 144, maxCount: 2 });

    assert.deepEqual(frames(next), [["a", 0], ["n1", 10]]);
  });
});
