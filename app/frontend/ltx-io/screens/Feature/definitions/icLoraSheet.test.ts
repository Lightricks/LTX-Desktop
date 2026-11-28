import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { getIcLoraSheetRecipe } from "../../../../lib/ic-lora-recipes.ts";
import { createIcLoraSheetDefinition } from "./icLoraSheet.ts";
import { specFixture } from "./specFixture.ts";

const definition = createIcLoraSheetDefinition(getIcLoraSheetRecipe("ingredients"));
const context = { specs: specFixture() };
const images = [{ assetId: "c" }, { assetId: "a" }, { assetId: "b" }];

describe("ingredients definition", () => {
  it("restores the images in the saved order, not sorted", () => {
    const restored = definition.fromGeneration(
      {
        params: {
          prompt: "p",
          model: "ltx-2.5-fast",
          aspectRatio: "9:16",
          resolution: "540p",
          duration: 8,
          fps: 25,
          audioMode: "off",
          loras: [{ scale: 1.2, variantId: "" }],
        },
        inputs: { images },
      },
      context,
    );
    assert.deepEqual(restored?.images, images);
    assert.equal(restored?.duration, 8);
    assert.equal(restored?.aspectRatio, "9:16");
  });

  it("sends the images in the list order and no video", () => {
    const body = definition.toCreateBody({ ...definition.defaults, images }, context);
    assert.deepEqual(body.inputs, { images });
  });

  it("reports prompt-required for a blank prompt even when images are present", () => {
    const issues = definition.validate({ ...definition.defaults, images, prompt: "   " }, context);
    assert.deepEqual(
      issues.map((issue) => issue.id),
      ["prompt-required"],
    );
  });

  it("is not ready with a blank prompt or with no images, so Generate stays off", () => {
    const ready = definition.normalize({ ...definition.defaults, images }, context);
    assert.equal(definition.isReady?.(ready, context), true);
    assert.equal(definition.isReady?.({ ...ready, prompt: "  " }, context), false);
    assert.equal(definition.isReady?.({ ...ready, images: [] }, context), false);
  });
});
