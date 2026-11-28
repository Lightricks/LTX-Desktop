import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  assetIdOf,
  enhanceImageAssetId,
  enhanceKeyframeFields,
  hasManualEnhanceInput,
  isEnhanceSourceCurrent,
  isPanelListCurrent,
  isSameReplacedWindow,
  isSourceVideoCurrent,
  keyframesSignature,
  lookImageAssetId,
  replacedWindowFor,
  resolveShowManualEnhance,
  resolveShowNagStrengthControls,
} from "./manualEnhanceVisibility.ts";

describe("resolveShowManualEnhance", () => {
  it("hides Enhance when Desktop auto-enhance is on", () => {
    assert.equal(resolveShowManualEnhance(true, true), false);
  });

  it("shows Enhance when Desktop auto-enhance is off", () => {
    assert.equal(resolveShowManualEnhance(false, false), true);
  });

  it("follows status fetched on feature enter when Remote has no live setting", () => {
    assert.equal(resolveShowManualEnhance(undefined, false), false);
    assert.equal(resolveShowManualEnhance(undefined, true), true);
  });
});

describe("hasManualEnhanceInput", () => {
  it("enables Enhance for a start frame or a source video with an empty prompt", () => {
    assert.equal(hasManualEnhanceInput({ prompt: "  ", startFrame: { assetId: "a" } }), true);
    assert.equal(hasManualEnhanceInput({ prompt: "", video: { assetId: "v" } }), true);
  });

  it("enables Enhance for the images of an image-list recipe, not for an empty list", () => {
    assert.equal(hasManualEnhanceInput({ prompt: " ", images: [{ assetId: "a" }] }), true);
    assert.equal(hasManualEnhanceInput({ prompt: " ", images: [] }), false);
  });

  it("enables Enhance for one still of a Keyframes clip with an empty prompt, not for none", () => {
    const still = { asset: { assetId: "a" }, frameIndex: 0, strength: 1 };
    assert.equal(hasManualEnhanceInput({ prompt: "", keyframes: [still] }), true);
    assert.equal(hasManualEnhanceInput({ prompt: "", keyframes: [] }), false);
  });

  it("enables Enhance for a typed prompt alone", () => {
    assert.equal(hasManualEnhanceInput({ prompt: "a fox" }), true);
  });

  it("keeps Enhance disabled when nothing it can read is present", () => {
    assert.equal(hasManualEnhanceInput({ prompt: " ", startFrame: null, video: null }), false);
  });
});

describe("isSourceVideoCurrent", () => {
  it("drops a rewrite built from a video the user has since replaced or removed", () => {
    assert.equal(isSourceVideoCurrent("new", "old"), false);
    assert.equal(isSourceVideoCurrent(null, "old"), false);
  });

  it("keeps a rewrite for the same video, for no video, and when the form values are gone", () => {
    assert.equal(isSourceVideoCurrent("same", "same"), true);
    assert.equal(isSourceVideoCurrent(null, null), true);
    assert.equal(isSourceVideoCurrent(undefined, "old"), true);
  });
});

describe("isPanelListCurrent", () => {
  it("drops a rewrite built from a list the user has since changed or reordered", () => {
    assert.equal(isPanelListCurrent(["a", "b", "c"], ["a", "b"]), false);
    assert.equal(isPanelListCurrent(["a"], ["a", "b"]), false);
    assert.equal(isPanelListCurrent(["b", "a"], ["a", "b"]), false);
  });

  it("keeps a rewrite for the same list, and when the form values are gone", () => {
    assert.equal(isPanelListCurrent(["a", "b"], ["a", "b"]), true);
    assert.equal(isPanelListCurrent(undefined, ["a", "b"]), true);
  });
});

describe("replacedWindowFor", () => {
  it("sends a Retake selection as the create body does, with the duration rounded the same way", () => {
    assert.deepEqual(replacedWindowFor("retake", { startTime: 2, duration: 4.724674999999999 }), {
      start: 2,
      end: 6.7,
    });
  });

  it("sends no window for any other feature, or when the form has no selection yet", () => {
    assert.equal(replacedWindowFor("extend", { startTime: 2, duration: 4 }), undefined);
    assert.equal(replacedWindowFor("text-to-video", { startTime: 2, duration: 4 }), undefined);
    assert.equal(replacedWindowFor("retake", {}), undefined);
    assert.equal(replacedWindowFor("retake", { startTime: 2 }), undefined);
  });
});

describe("isSameReplacedWindow", () => {
  it("is false when the Retake start or end moved, so a result from the old window is dropped", () => {
    const sent = { start: 2, end: 6 };
    assert.equal(isSameReplacedWindow({ start: 3, end: 6 }, sent), false);
    assert.equal(isSameReplacedWindow({ start: 2, end: 7 }, sent), false);
  });

  it("is true for the same window and when neither side has one", () => {
    assert.equal(isSameReplacedWindow({ start: 2, end: 6 }, { start: 2, end: 6 }), true);
    assert.equal(isSameReplacedWindow(undefined, undefined), true);
  });

  it("is false when only one side has a window", () => {
    assert.equal(isSameReplacedWindow(undefined, { start: 2, end: 6 }), false);
    assert.equal(isSameReplacedWindow({ start: 2, end: 6 }, undefined), false);
  });
});

describe("lookImageAssetId", () => {
  const video = { assetId: "video-1" };
  const image = { assetId: "image-1" };

  it("sends the look image of an IC-LoRA recipe that has a video", () => {
    assert.equal(lookImageAssetId({ isIcLoraRecipe: true, video, image }), "image-1");
  });

  it("sends no image for an IC-LoRA driven by an image, with no video", () => {
    assert.equal(lookImageAssetId({ isIcLoraRecipe: true, video: null, image }), undefined);
  });

  it("sends no image for a feature that is not an IC-LoRA recipe", () => {
    assert.equal(lookImageAssetId({ isIcLoraRecipe: false, video, image }), undefined);
  });

  it("sends none when the optional look image is not set", () => {
    assert.equal(lookImageAssetId({ isIcLoraRecipe: true, video, image: null }), undefined);
  });

  it("sends the start frame for Motion Track and drops a stale result when startFrame changes", () => {
    assert.equal(
      enhanceImageAssetId({
        featureId: "motion-track",
        isIcLoraRecipe: true,
        startFrame: { assetId: "frog" },
      }),
      "frog",
    );
    assert.equal(isSourceVideoCurrent("cat", "frog"), false);
    assert.equal(isSourceVideoCurrent("frog", "frog"), true);
  });
});

describe("assetIdOf", () => {
  it("reads null for a value with no string assetId, so a number or a missing key is not an id", () => {
    assert.equal(assetIdOf({ assetId: 7 }), null);
    assert.equal(assetIdOf({ id: "frog" }), null);
    assert.equal(assetIdOf(null), null);
    assert.equal(assetIdOf({ assetId: "frog" }), "frog");
  });
});

describe("enhanceImageAssetId", () => {
  it("sends the start frame for Image to Video", () => {
    assert.equal(
      enhanceImageAssetId({
        featureId: "image-to-video",
        isIcLoraRecipe: false,
        startFrame: { assetId: "frog" },
      }),
      "frog",
    );
  });

  it("sends none for a video IC-LoRA with no source video", () => {
    assert.equal(
      enhanceImageAssetId({
        featureId: "day-to-night",
        isIcLoraRecipe: true,
        image: { assetId: "look" },
      }),
      undefined,
    );
  });
});

describe("keyframesSignature", () => {
  const still = (assetId: string, frameIndex: number, strength = 1) => ({
    asset: { assetId },
    frameIndex,
    strength,
  });

  it("does not depend on the order the stills were added in", () => {
    assert.equal(
      keyframesSignature([still("b", 48), still("a", 0)]),
      keyframesSignature([still("a", 0), still("b", 48)]),
    );
  });

  it("changes when a strength changes, because the rewrite weights each still by it", () => {
    assert.notEqual(keyframesSignature([still("a", 0, 1)]), keyframesSignature([still("a", 0, 0.4)]));
  });

  it("changes when the clip length or the frame rate changes, even if no frame moves", () => {
    const stills = [still("a", 0)];
    const base = keyframesSignature(stills, { duration: 5, fps: 24 });
    assert.notEqual(base, keyframesSignature(stills, { duration: 6, fps: 24 }));
    assert.notEqual(base, keyframesSignature(stills, { duration: 5, fps: 48 }));
    assert.equal(base, keyframesSignature(stills, { duration: 5, fps: 24 }));
  });

  it("is empty for no stills, whatever the clip timing is", () => {
    assert.equal(keyframesSignature([], { duration: 5, fps: 24 }), "");
    assert.equal(keyframesSignature(null, { duration: 5, fps: 24 }), "");
  });
});

describe("enhanceKeyframeFields", () => {
  const still = (assetId: string, frameIndex: number) => ({
    asset: { assetId },
    frameIndex,
    strength: 0.5,
  });

  it("sends the stills in time order with the clip timing", () => {
    assert.deepEqual(
      enhanceKeyframeFields({ keyframes: [still("b", 48), still("a", 0)], duration: 5, fps: 24 }),
      {
        keyframeAssets: [
          { assetId: "a", frameIndex: 0, strength: 0.5 },
          { assetId: "b", frameIndex: 48, strength: 0.5 },
        ],
        duration: 5,
        fps: 24,
      },
    );
  });

  it("sends no timing the backend would refuse, and nothing for a form without stills", () => {
    assert.deepEqual(enhanceKeyframeFields({ keyframes: [still("a", 0)], duration: 5.5, fps: 24 })?.duration, undefined);
    assert.equal(enhanceKeyframeFields({ keyframes: [], duration: 5, fps: 24 }), undefined);
  });
});

describe("isEnhanceSourceCurrent", () => {
  const source = {
    videoAssetId: null,
    startFrameAssetId: "frog",
    referenceImageAssetId: null,
    panelAssetIds: [],
    keyframes: "",
    replacedWindow: undefined,
  };
  const live = { ...source };

  it("drops a rewrite of old keyframes, so it does not land on a changed timeline", () => {
    const withStills = { ...source, keyframes: "a@0,b@48" };
    assert.equal(isEnhanceSourceCurrent({ ...withStills, keyframes: "a@0,b@72" }, withStills), false);
    assert.equal(isEnhanceSourceCurrent(withStills, withStills), true);
  });

  it("drops a rewrite of an old start frame, so it does not land on the new one", () => {
    assert.equal(isEnhanceSourceCurrent({ ...live, startFrameAssetId: "cat" }, source), false);
  });

  it("applies the rewrite when the start frame did not change", () => {
    assert.equal(isEnhanceSourceCurrent(live, source), true);
  });

  it("applies the rewrite when the live form values are gone", () => {
    assert.equal(
      isEnhanceSourceCurrent(
        {
          videoAssetId: undefined,
          startFrameAssetId: undefined,
          referenceImageAssetId: undefined,
          panelAssetIds: undefined,
          keyframes: undefined,
          replacedWindow: undefined,
        },
        source,
      ),
      true,
    );
  });
});


describe("resolveShowNagStrengthControls", () => {
  it("follows the live Desktop setting, even when the status is older", () => {
    assert.equal(resolveShowNagStrengthControls(true, false), true);
    assert.equal(resolveShowNagStrengthControls(false, true), false);
  });

  it("follows the status on Remote, which has no live setting", () => {
    assert.equal(resolveShowNagStrengthControls(undefined, true), true);
    assert.equal(resolveShowNagStrengthControls(undefined, false), false);
  });

  it("is off while the status has not loaded", () => {
    assert.equal(resolveShowNagStrengthControls(undefined, undefined), false);
  });
});
