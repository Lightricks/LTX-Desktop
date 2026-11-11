import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { FeatureKeyframe } from "../types.ts";
import type { KeyframesExampleSeed } from "./keyframesExample.ts";
import { specFixture } from "./specFixture.ts";
import {
  KEYFRAMES_DEFAULTS,
  fromGeneration,
  keyframesDefinition,
  seedKeyframesValues,
  toCreateBody,
  validateKeyframes,
  type KeyframesContext,
  type KeyframesValues,
} from "./keyframes.ts";

const context: KeyframesContext = { specs: specFixture() };

function keyframe(id: string, frameIndex: number, strength = 0.7): FeatureKeyframe {
  return { id, asset: { assetId: `asset-${id}` }, frameIndex, strength };
}

function values(overrides: Partial<KeyframesValues> = {}): KeyframesValues {
  return {
    ...KEYFRAMES_DEFAULTS,
    prompt: "a fox runs",
    keyframes: [keyframe("a", 0), keyframe("b", 96)],
    ...overrides,
  };
}

describe("keyframes definition", () => {
  it("moves the stills in when a shorter length leaves them past the clip end", () => {
    // 8 s at 24 fps ends on frame 192. 4 s ends on frame 96.
    const next = keyframesDefinition.applyChange(
      values({ keyframes: [keyframe("a", 0), keyframe("b", 192)] }),
      { kind: "options", fieldId: "duration", dataKey: "duration", value: 4 },
      context,
    );

    assert.equal(next.duration, 4);
    assert.deepEqual(
      next.keyframes.map((item) => item.frameIndex),
      [0, 96],
    );
  });

  it("sends the stills in time order, with no local id", () => {
    const body = toCreateBody(
      values({ keyframes: [keyframe("late", 96, 0.4), keyframe("early", 0, 1)] }),
      context,
    );

    assert.deepEqual(body.inputs.keyframes, [
      { assetId: "asset-early", frameIndex: 0, strength: 1 },
      { assetId: "asset-late", frameIndex: 96, strength: 0.4 },
    ]);
  });

  it("blocks an empty list, a blank prompt, and two stills on one frame", () => {
    const ids = (input: KeyframesValues) =>
      validateKeyframes(input, context)
        .filter((issue) => issue.blocksGenerate !== false)
        .map((issue) => issue.id);

    assert.deepEqual(ids(values({ keyframes: [] })), ["keyframes-required"]);
    assert.deepEqual(ids(values({ prompt: "  " })), ["prompt-required"]);
    assert.deepEqual(
      ids(values({ keyframes: [keyframe("a", 5), keyframe("b", 5)] })),
      ["keyframes-frames-invalid"],
    );
    assert.deepEqual(ids(values()), []);
  });

  it("does not throw on a past run whose keyframe list holds an empty item", () => {
    const restored = fromGeneration(
      {
        params: { prompt: "a fox runs", model: "ltx-2.5-fast", duration: 5, fps: 24 },
        inputs: { keyframes: [null] },
      },
      context,
    );

    assert.equal(restored, null);
  });

  it("restores the frame and strength of each still from a past run", () => {
    const restored = fromGeneration(
      {
        params: { prompt: "a fox runs", model: "ltx-2.5-fast", duration: 5, fps: 24 },
        inputs: {
          keyframes: [
            { assetId: "one", frameIndex: 0, strength: 0.25 },
            { assetId: "two", frameIndex: 120, strength: 1 },
          ],
        },
      },
      context,
    );

    assert.deepEqual(
      restored?.keyframes.map(({ asset, frameIndex, strength }) => ({
        assetId: asset.assetId,
        frameIndex,
        strength,
      })),
      [
        { assetId: "one", frameIndex: 0, strength: 0.25 },
        { assetId: "two", frameIndex: 120, strength: 1 },
      ],
    );
  });
});

describe("keyframes example seed", () => {
  const seed: KeyframesExampleSeed = {
    "keyframes-summer-image": { assetId: "summer" },
    "keyframes-autumn-image": { assetId: "autumn" },
    "keyframes-winter-image": { assetId: "winter" },
    "keyframes-spring-image": { assetId: "spring" },
  };

  it("opens on the four seasons, with summer at both ends of the 6 s clip", () => {
    const seeded = seedKeyframesValues({ ...context, seed });

    assert.deepEqual(
      seeded.keyframes.map(({ asset, frameIndex }) => [asset.assetId, frameIndex]),
      [
        ["summer", 0],
        ["autumn", 36],
        ["winter", 72],
        ["spring", 108],
        ["summer", 144],
      ],
    );
    assert.equal(seeded.duration, 6);
    assert.equal(seeded.fps, 24);
    assert.equal(seeded.resolution, "720p");
    assert.notEqual(seeded.prompt, "");
    assert.notEqual(seeded.negativePrompt, "");
    assert.equal(new Set(seeded.keyframes.map(({ id }) => id)).size, 5);
  });

  it("opens empty without the seed, and Reset never brings the example back", () => {
    assert.deepEqual(seedKeyframesValues(context), keyframesDefinition.resetValues?.(context));
    assert.deepEqual(keyframesDefinition.resetValues?.({ ...context, seed })?.keyframes, []);
  });
});
