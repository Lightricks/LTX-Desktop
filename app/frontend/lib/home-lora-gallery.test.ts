import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { HOME_LORA_GALLERY_SECTIONS } from "./home-lora-gallery.ts";
import { LORA_RECIPES } from "./lora-recipes.ts";

describe("home lora gallery", () => {
  it("lists every recipe once", () => {
    const listed = HOME_LORA_GALLERY_SECTIONS.flatMap(
      (section) => section.featureIds,
    );
    assert.equal(new Set(listed).size, listed.length);
    assert.deepEqual(
      [...listed].sort(),
      LORA_RECIPES.map((recipe) => recipe.id).sort(),
    );
  });
});
