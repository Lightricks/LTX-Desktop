import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { getIcLoraRecipe, type IcLoraRecipe } from "../../lib/ic-lora-recipes.ts";
import { planIcLoraImageSeed } from "./icLoraRecipeSeed.ts";

const recipe = getIcLoraRecipe("layout-to-render");
const PACKAGED_NAME = "layout-to-render-first-frame.jpg";

type ImageSeedInput = Parameters<typeof planIcLoraImageSeed>[1];

function plan(overrides: Partial<ImageSeedInput> = {}, target: IcLoraRecipe = recipe) {
  return planIcLoraImageSeed(target, {
    hasStoredValues: true,
    storedImageId: "stored-image",
    generationsReady: true,
    generationsFailed: false,
    lastGenerationSpec: undefined,
    hydratedImageName: PACKAGED_NAME,
    hydratedImagePending: false,
    appliedSeedRevision: 1,
    ...overrides,
  });
}

/** The same recipe after the packaged look image was replaced: revision 2, old name kept. */
function withBumpedImageSeed(): IcLoraRecipe {
  const image = recipe.referenceImage;
  assert.ok(image?.seed);
  return {
    ...recipe,
    referenceImage: {
      ...image,
      seed: {
        ...image.seed,
        filename: "layout-to-render-first-frame-v2.jpg",
        previousFilenames: [PACKAGED_NAME],
        revision: 2,
      },
    },
  };
}

describe("planIcLoraImageSeed", () => {
  it("keeps the stored image, even a user image with the packaged filename, once the revision is applied", () => {
    assert.equal(plan(), "skip");
  });

  it("replaces the stored packaged image when the applied revision is older than the packaged one", () => {
    assert.equal(plan({ appliedSeedRevision: 0 }), "ingest");
    // A bumped packaged image replaces the copy that carries the previous filename.
    assert.equal(plan({ appliedSeedRevision: 1 }, withBumpedImageSeed()), "ingest");
  });

  it("keeps a stored image with another filename when the applied revision is older", () => {
    assert.equal(plan({ appliedSeedRevision: 0, hydratedImageName: "my-photo.jpg" }), "skip");
  });

  it("ingests the packaged image for stored values with an empty image before the first apply", () => {
    // The race: the form was stored before the packaged image loaded.
    const raced = { storedImageId: null, hydratedImageName: null, appliedSeedRevision: 0 };
    assert.equal(plan(raced), "ingest");
    assert.equal(plan({ ...raced, generationsReady: false }), "wait");
  });

  it("keeps a cleared image cleared once the packaged revision was applied", () => {
    assert.equal(
      plan({ storedImageId: null, hydratedImageName: null, appliedSeedRevision: 1 }),
      "skip",
    );
  });

  it("never plans an image for a recipe with no packaged image", () => {
    assert.equal(plan({ appliedSeedRevision: 0 }, getIcLoraRecipe("day-to-night")), "skip");
  });
});
