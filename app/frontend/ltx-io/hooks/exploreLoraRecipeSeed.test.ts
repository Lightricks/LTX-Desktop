import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { LORA_RECIPES, recipeHasEndFrame } from "../../lib/lora-recipes.ts";

import {
  LORA_RECIPE_END_FRAME_SEEDS,
  LORA_RECIPE_START_FRAME_SEEDS,
  planExploreLoraRecipeSeed,
} from "./exploreLoraRecipeSeed.ts";

const hydratingSpec = {
  params: { prompt: "from history", aspectRatio: "auto" },
  inputs: { startFrame: { assetId: "history-start" } },
};

describe("planExploreLoraRecipeSeed", () => {
  it("skips ingest for t2v recipes", () => {
    assert.equal(
      planExploreLoraRecipeSeed({
        mode: "t2v",
        hasStoredValues: false,
        generationsReady: true,
        generationsFailed: false,
        lastGenerationSpec: undefined,
      }),
      "skip",
    );
  });

  it("skips ingest when persisted values already exist", () => {
    assert.equal(
      planExploreLoraRecipeSeed({
        mode: "i2v",
        hasStoredValues: true,
        generationsReady: false,
        generationsFailed: false,
        lastGenerationSpec: undefined,
      }),
      "skip",
    );
  });

  it("skips ingest when a durable generation can restore Start Frame", () => {
    assert.equal(
      planExploreLoraRecipeSeed({
        mode: "i2v",
        hasStoredValues: false,
        generationsReady: true,
        generationsFailed: false,
        lastGenerationSpec: hydratingSpec,
      }),
      "skip",
    );
  });

  it("waits for generations before ingesting on a first visit", () => {
    assert.equal(
      planExploreLoraRecipeSeed({
        mode: "i2v",
        hasStoredValues: false,
        generationsReady: false,
        generationsFailed: false,
        lastGenerationSpec: undefined,
      }),
      "wait",
    );
  });

  it("ingests the packaged seed when nothing durable can hydrate Start Frame", () => {
    assert.equal(
      planExploreLoraRecipeSeed({
        mode: "i2v",
        hasStoredValues: false,
        generationsReady: true,
        generationsFailed: false,
        lastGenerationSpec: undefined,
      }),
      "ingest",
    );
  });

  it("ingests when last generation has a prompt but no start frame", () => {
    assert.equal(
      planExploreLoraRecipeSeed({
        mode: "i2v",
        hasStoredValues: false,
        generationsReady: true,
        generationsFailed: false,
        lastGenerationSpec: { params: { prompt: "old t2v row" } },
      }),
      "ingest",
    );
  });

  it("maps every i2v recipe to a packaged start-frame seed", () => {
    for (const recipe of LORA_RECIPES) {
      if (recipe.mode === "i2v") {
        assert.ok(
          LORA_RECIPE_START_FRAME_SEEDS[recipe.id],
          `${recipe.id} is i2v but has no packaged start-frame seed`,
        );
      }
    }
    assert.equal(LORA_RECIPE_START_FRAME_SEEDS["dolly-in"], "dolly-in-start-frame");
    for (const recipe of LORA_RECIPES) {
      if (recipeHasEndFrame(recipe)) {
        assert.ok(
          LORA_RECIPE_END_FRAME_SEEDS[recipe.id],
          `${recipe.id} requires End Frame but has no packaged end-frame seed`,
        );
      }
    }
    assert.equal(LORA_RECIPE_END_FRAME_SEEDS.transition, "transition-end-frame");
  });

  it("does not ingest when generation listing failed", () => {
    assert.equal(
      planExploreLoraRecipeSeed({
        mode: "i2v",
        hasStoredValues: false,
        generationsReady: false,
        generationsFailed: true,
        lastGenerationSpec: undefined,
      }),
      "skip",
    );
  });
});
