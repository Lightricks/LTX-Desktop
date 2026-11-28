import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { orderAspectRatiosForPicker } from "../../../lib/video-generation-model-specs.ts";
import {
  IMAGE_ASPECTS,
  aspectRatioNumber,
  fittingImageResolution,
  imageAspectForVideo,
  imageResolutionFits,
  imageSizeForAspect,
} from "./imageGeneratorRequest.ts";

describe("image size", () => {
  it("the resolution is the short side, so 1080p at 16:9 is 1920x1080 as in GenSpace", () => {
    assert.deepEqual(imageSizeForAspect("16:9", "1080p"), { width: 1920, height: 1080 });
    assert.deepEqual(imageSizeForAspect("9:16", "1080p"), { width: 1080, height: 1920 });
    assert.deepEqual(imageSizeForAspect("4:3", "1440p"), { width: 1920, height: 1440 });
  });

  it("a size over the backend limit does not fit, so the request is never sent", () => {
    assert.equal(imageResolutionFits("16:9", "1440p"), false);
    assert.equal(imageResolutionFits("1:1", "2048p"), true);
    assert.equal(imageResolutionFits("4:3", "2048p"), false);
  });

  it("a shape that cannot fit the chosen resolution falls back to the largest one that fits", () => {
    assert.equal(fittingImageResolution("1:1", "2048p"), "2048p");
    assert.equal(fittingImageResolution("4:3", "2048p"), "1440p");
    assert.equal(fittingImageResolution("16:9", "2048p"), "1080p");
  });

  it("a smaller choice is never raised, so 1080p stays 1080p on a square", () => {
    assert.equal(fittingImageResolution("1:1", "1080p"), "1080p");
  });
});

describe("preview frame ratio", () => {
  it("a tall shape gives a ratio under 1, so the empty frame is tall before the image exists", () => {
    assert.equal(aspectRatioNumber("9:16"), 9 / 16);
    assert.equal(aspectRatioNumber("1:1"), 1);
  });
});

describe("image aspect ratio order", () => {
  it("the picker lists the shapes in the order of the feature forms, so the two pickers match", () => {
    assert.deepEqual(orderAspectRatiosForPicker(IMAGE_ASPECTS), [...IMAGE_ASPECTS]);
    assert.deepEqual(IMAGE_ASPECTS.slice(0, 2), ["16:9", "9:16"]);
  });
});

describe("image shape for a video aspect ratio", () => {
  it("a 16:9 and a 9:16 video give the same image shape, so the frame fills the video", () => {
    assert.equal(imageAspectForVideo("16:9"), "16:9");
    assert.equal(imageAspectForVideo("9:16"), "9:16");
  });

  it("a video ratio the image picker lacks gives the nearest shape, so 4:5 is not a landscape image", () => {
    assert.equal(imageAspectForVideo("4:5"), "3:4");
    assert.equal(imageAspectForVideo("3:2"), "4:3");
    assert.equal(imageAspectForVideo("21:9"), "16:9");
    assert.equal(imageAspectForVideo("1:1"), "1:1");
  });

  it("auto gives no shape, so the generator keeps its own default of 16:9", () => {
    assert.equal(imageAspectForVideo("auto"), undefined);
  });
});
