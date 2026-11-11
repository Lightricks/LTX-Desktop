import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import { createIcLoraRecipeDefinition } from "../ltx-io/screens/Feature/definitions/icLoraRecipe.ts";
import { createIcLoraSheetDefinition } from "../ltx-io/screens/Feature/definitions/icLoraSheet.ts";
import { createIcLoraTrackDefinition } from "../ltx-io/screens/Feature/definitions/icLoraTrack.ts";
import { getHomeFeature } from "./home-features.ts";
import {
  getIcLoraRecipe,
  IC_LORA_RECIPES,
  IC_LORA_SHEET_MAX_IMAGES,
  isIcLoraRecipeId,
  type IcLoraRecipe,
  type IcLoraRecipeId,
} from "./ic-lora-recipes.ts";

type RecipeContractEntry = {
  id: string;
  catalogId: string;
  kind: string;
  highResWithoutFlag?: true;
};

/**
 * Rows whose example and Home card media are still copies of the Restore media. Delete
 * the id here when the row has its own media under frontend/ltx-io/assets/<id>/.
 */
const PLACEHOLDER_EXAMPLE_MEDIA: ReadonlySet<string> = new Set(["refine-details"]);

const SHARED_CONTRACT: RecipeContractEntry[] = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("../../shared/ic-lora-recipes.json", import.meta.url)),
    "utf8",
  ),
);

const repoRoot = fileURLToPath(new URL("../..", import.meta.url));

const SHIPPED_CATALOG: { ic_loras: { id: string; input?: { max_count?: number } }[] } = JSON.parse(
  readFileSync(`${repoRoot}/backend/runtime_config/lora_catalog.json`, "utf8"),
);

describe("ic-lora recipe registry", () => {
  it("matches the shared id, catalogId and kind contract", () => {
    assert.deepEqual(
      IC_LORA_RECIPES.map((recipe) => ({
        id: recipe.id,
        catalogId: recipe.catalogId,
        kind: recipe.kind,
        ...("highResWithoutFlag" in recipe && recipe.highResWithoutFlag === true
          ? { highResWithoutFlag: true }
          : {}),
      })),
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
      assert.equal(getHomeFeature(id).id, recipe.id);
      switch (recipe.kind) {
        case "video":
          assert.equal(createIcLoraRecipeDefinition(recipe).id, recipe.id);
          assert.equal(
            existsSync(`${repoRoot}/resources/explore-assets/${recipe.seed.filename}`),
            true,
            recipe.seed.filename,
          );
          break;
        case "sheet":
          assert.equal(createIcLoraSheetDefinition(recipe).id, recipe.id);
          break;
        case "track":
          assert.equal(createIcLoraTrackDefinition(recipe).id, recipe.id);
          break;
        default: {
          const unreachable: never = recipe;
          throw new Error(`Unhandled recipe kind: ${JSON.stringify(unreachable)}`);
        }
      }
      if (PLACEHOLDER_EXAMPLE_MEDIA.has(recipe.id)) continue;
      assert.equal(
        existsSync(
          `${repoRoot}/frontend/ltx-io/assets/${recipe.id}/example.${recipe.kind === "track" ? "webm" : "mp4"}`,
        ),
        true,
      );
      assert.equal(
        existsSync(`${repoRoot}/frontend/ltx-io/assets/${recipe.id}/example-poster.webp`),
        true,
      );
    }
  });

  it("offers as many images as the shipped Ingredients catalog entry accepts", () => {
    const ingredients = SHIPPED_CATALOG.ic_loras.find((entry) => entry.id === "ingredients");
    assert.equal(ingredients?.input?.max_count, IC_LORA_SHEET_MAX_IMAGES);
  });

  it("gives a reference image to Layout to Render, Restore and Video Guide only", () => {
    const recipes: readonly IcLoraRecipe[] = IC_LORA_RECIPES;
    assert.deepEqual(
      recipes.filter((recipe) => recipe.referenceImage != null).map((recipe) => recipe.id),
      ["layout-to-render", "restore", "video-guide"],
    );
  });
});
