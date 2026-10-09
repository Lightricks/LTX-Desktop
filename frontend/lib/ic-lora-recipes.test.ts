import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import { createIcLoraRecipeDefinition } from "../ltx-io/screens/Feature/definitions/icLoraRecipe.ts";
import { getHomeFeature } from "./home-features.ts";
import {
  getIcLoraRecipe,
  IC_LORA_RECIPES,
  isIcLoraRecipeId,
  type IcLoraRecipe,
  type IcLoraRecipeId,
} from "./ic-lora-recipes.ts";

type RecipeContractEntry = {
  id: string;
  catalogId: string;
};

const SHARED_CONTRACT: RecipeContractEntry[] = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("../../shared/ic-lora-recipes.json", import.meta.url)),
    "utf8",
  ),
);

const repoRoot = fileURLToPath(new URL("../..", import.meta.url));

describe("ic-lora recipe registry", () => {
  it("matches the shared id and catalogId contract", () => {
    assert.deepEqual(
      IC_LORA_RECIPES.map((recipe) => ({ id: recipe.id, catalogId: recipe.catalogId })),
      SHARED_CONTRACT,
    );
    assert.equal(isIcLoraRecipeId("day-to-night"), true);
    assert.equal(getIcLoraRecipe("day-to-night").catalogId, "day-to-night");
    assert.equal(isIcLoraRecipeId("alpha-gen"), true);
    assert.equal(getIcLoraRecipe("alpha-gen").catalogId, "alpha-gen");
    assert.equal(isIcLoraRecipeId("layout-to-render"), true);
    assert.equal(getIcLoraRecipe("layout-to-render").catalogId, "layout-to-render");
  });

  it("wires every row to a screen, a definition, a seed file, and example media", () => {
    for (const recipe of IC_LORA_RECIPES) {
      const id = recipe.id as IcLoraRecipeId;
      assert.equal(createIcLoraRecipeDefinition(recipe).id, recipe.id);
      assert.equal(getHomeFeature(id).id, recipe.id);
      assert.equal(
        existsSync(`${repoRoot}/resources/explore-assets/${recipe.seed.filename}`),
        true,
        recipe.seed.filename,
      );
      assert.equal(
        existsSync(`${repoRoot}/frontend/ltx-io/assets/${recipe.id}/example.mp4`),
        true,
      );
      assert.equal(
        existsSync(`${repoRoot}/frontend/ltx-io/assets/${recipe.id}/example-poster.webp`),
        true,
      );
    }
  });

  it("gives a reference image to Layout to Render and Restore only", () => {
    const recipes: readonly IcLoraRecipe[] = IC_LORA_RECIPES;
    assert.deepEqual(
      recipes.filter((recipe) => recipe.referenceImage != null).map((recipe) => recipe.id),
      ["layout-to-render", "restore"],
    );
  });
});
