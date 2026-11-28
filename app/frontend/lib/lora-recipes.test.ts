import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import { paths } from "../paths.ts";
import {
  getLoraRecipe,
  isLoraRecipeId,
  LORA_RECIPES,
  LORA_RECIPE_HOME_FEATURES,
  recipeHasEndFrame,
} from "./lora-recipes.ts";

type RecipeContractEntry = {
  id: string;
  catalogId: string;
  mode: string;
  devices: string[];
  listed: boolean;
  requiresEndFrame?: boolean;
};

// shared/lora-recipes.json is the single source of truth for the id / catalogId /
// mode / devices / listed contract. Both this FE table and the backend registry
// (test_registry_matches_shared_contract) are asserted against it, so a change on
// one side that isn't mirrored in the JSON fails a test rather than only surfacing
// at runtime as a LORA_UNKNOWN create error. Read via fs (not a JSON import) so it
// works under the node test runner.
const SHARED_CONTRACT: RecipeContractEntry[] = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("../../shared/lora-recipes.json", import.meta.url)),
    "utf8",
  ),
);

describe("lora recipe registry", () => {
  it("matches the shared FE/BE contract (id -> catalogId)", () => {
    const byId = new Map(SHARED_CONTRACT.map((entry) => [entry.id, entry]));
    for (const recipe of LORA_RECIPES) {
      const entry = byId.get(recipe.id);
      assert.ok(entry, `recipe ${recipe.id} missing from shared/lora-recipes.json`);
      assert.equal(recipe.catalogId, entry.catalogId);
      assert.equal(recipe.listed, entry.listed);
      assert.equal(recipe.mode, entry.mode);
      assert.equal(recipeHasEndFrame(recipe), Boolean(entry.requiresEndFrame));
    }
    const recipe = getLoraRecipe("cozy-felt");
    assert.equal(recipe.catalogId, "cozy-felt-style");
    assert.equal(recipe.path, paths.cozyFelt);
    assert.equal(recipe.listing.typeLabel, "Text to Video");
    assert.match(recipe.listing.blurb, /handcrafted felt world/);
    const dollyIn = getLoraRecipe("dolly-in");
    assert.equal(dollyIn.listing.typeLabel, "Image to Video");
    assert.equal(dollyIn.listing.affiliation, "ltx");
  });

  it("derives home-feature rows from the listed recipes only", () => {
    const listedIds = SHARED_CONTRACT.filter((entry) => entry.listed).map(
      (entry) => entry.id,
    );
    assert.deepEqual(
      LORA_RECIPE_HOME_FEATURES.map((feature) => feature.id),
      listedIds,
    );
    for (const feature of LORA_RECIPE_HOME_FEATURES) {
      const recipe = getLoraRecipe(feature.id);
      assert.equal(feature.title, recipe.title);
      assert.equal(feature.description, recipe.description);
      assert.equal(feature.path, recipe.path);
      assert.equal(feature.fetcherTool, recipe.id);
    }
  });

  it("recognizes only registered recipe ids", () => {
    assert.equal(isLoraRecipeId("cozy-felt"), true);
    assert.equal(isLoraRecipeId("dolly-in"), true);
    assert.equal(isLoraRecipeId("not-a-recipe"), false);
    assert.equal(isLoraRecipeId(undefined), false);
  });

  it("prefills dolly-in with the ltx.io example scene, not an empty prompt", () => {
    // Matches infinity/src/utils/loraSeedPrompts.ts LORA_PROMPTS_BY_ID["dolly-in"].
    const recipe = getLoraRecipe("dolly-in");
    assert.equal(recipe.catalogId, "dolly-in");
    assert.equal(recipe.mode, "i2v");
    assert.equal(recipe.path, paths.dollyIn);
    assert.match(recipe.seedPrompt, /camera slowly zooms in on the character's face/i);
  });

  it("prefills cozy-felt with the ltx.io example scene, not an empty prompt", () => {
    // Matches infinity/src/utils/loraSeedPrompts.ts LORA_PROMPTS_BY_ID["cozy-felt"].
    // An empty seed leaves the textarea blank (placeholder-only); the form default
    // is recipe.seedPrompt, so this is what the user sees on first open.
    const recipe = getLoraRecipe("cozy-felt");
    assert.match(recipe.seedPrompt, /felt T-rex plays golf/i);
  });

  it("marks only Transition as requiring an end frame", () => {
    const transition = getLoraRecipe("transition");
    assert.equal(transition.mode, "i2v");
    assert.equal(
      "requiresEndFrame" in transition && transition.requiresEndFrame,
      true,
    );
    assert.equal(getLoraRecipe("dolly-in").mode, "i2v");
    assert.equal("requiresEndFrame" in getLoraRecipe("dolly-in"), false);
  });

  it("puts FeatureDetails listing copy on every recipe row", () => {
    for (const recipe of LORA_RECIPES) {
      assert.equal(
        recipe.listing.typeLabel,
        recipe.mode === "i2v" ? "Image to Video" : "Text to Video",
        recipe.id,
      );
      assert.ok(recipe.listing.blurb.length > 0, recipe.id);
      assert.ok(recipe.listing.huggingfaceUrl.startsWith("https://huggingface.co/"), recipe.id);
      assert.ok(
        recipe.listing.affiliation === "ltx" ||
          recipe.listing.affiliation === "community",
        recipe.id,
      );
    }
  });
});
