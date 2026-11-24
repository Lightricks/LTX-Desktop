import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { components } from "../../../../generated/backend-openapi.ts";
import { getIcLoraRecipe } from "../../../../lib/ic-lora-recipes.ts";
import controls from "../../../../../shared/ic-lora-tiling-controls.json" with { type: "json" };
import { createIcLoraRecipeDefinition, type IcLoraRecipeContext } from "./icLoraRecipe.ts";
import { tileSizeOptions } from "./icLoraTileOptions.ts";
import { AUTO_TILE, TILE_CHOICES, TILE_PRESETS } from "./icLoraTilingControls.ts";
import { specFixture } from "./specFixture.ts";

type TilePreflight = components["schemas"]["IcLoraTilePreflightResponse"];
type TileCell = components["schemas"]["IcLoraTilePreflightCell"];

const REFUSAL = "This job needs more VRAM than this machine has.";

function preflight(
  cells: Partial<Record<TileCell["tileSize"], Partial<TileCell>>>,
  budgetMode: TilePreflight["budgetMode"] = "cautious",
): TilePreflight {
  return {
    budgetMode,
    cells: (["standard", "large", "extra-large"] as const).map((tileSize) => ({
      tileSize,
      ok: true,
      ...cells[tileSize],
    })),
    defaults: { carryFrames: 25, auto: { kind: "windowed", frames: 97 } },
  };
}

function context(tilePreflight?: TilePreflight): IcLoraRecipeContext {
  return {
    specs: specFixture(),
    seed: null,
    videoDurationSeconds: 4,
    videoDurationPending: false,
    videoWidth: 1920,
    videoHeight: 1080,
    videoFps: null,
    tilingAdaptive: true,
    tilingTile: { long: 1024, short: 576 },
    tilePreflight,
  };
}

const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("refine-details"));
const withVideo = { ...definition.defaults, video: { assetId: "clip" } };

describe("refine details tile size", () => {
  it("uses the tile names from the shared controls file, plus Standard for the catalog tile", () => {
    assert.deepEqual(
      TILE_PRESETS.filter((name) => name !== "standard").slice().sort(),
      Object.keys(controls.tileSizes).sort(),
    );
    assert.deepEqual([...TILE_CHOICES].sort(), Object.keys(controls.tileSizes).sort());
  });

  it("offers Auto first and labels it with the catalog tile it runs", () => {
    const options = tileSizeOptions(context());
    assert.deepEqual(
      options.map((option) => option.value),
      [AUTO_TILE, "large", "extra-large"],
    );
    assert.equal(options[0]?.label, "Auto (Standard 1024\u00d7576)");
  });

  it("disables a size the budget refuses and carries the refusal as the reason", () => {
    const options = tileSizeOptions(
      context(preflight({ "extra-large": { ok: false, message: REFUSAL } })),
    );
    assert.deepEqual(
      options.map((option) => option.disabledReason),
      [undefined, undefined, REFUSAL],
    );
    assert.equal(options[0]?.disabledReason, undefined, "Auto is never disabled");
  });

  it("keeps an over-budget size that Brave admits a plain option with no warning and no badge", () => {
    const readout = { kind: "gpu", need_gb: 14, have_gb: 12 } as const;
    const options = tileSizeOptions(
      context(
        preflight(
          { large: { ok: true, tier: "over", message: "Tight on VRAM.", readout } },
          "brave",
        ),
      ),
    );
    const large = options.find((option) => option.value === "large");
    assert.equal(large?.disabledReason, undefined);
    assert.equal(large?.warning, undefined);
    assert.equal(large?.budget, undefined);
  });

  it("leaves every size enabled while the verdicts are unknown", () => {
    const options = tileSizeOptions(context());
    assert.deepEqual(
      options.map((option) => option.disabledReason),
      [undefined, undefined, undefined],
    );
  });

  it("resets a refused selection to Auto and keeps an allowed one", () => {
    const selected = { ...withVideo, tilingSize: "extra-large" as const };
    const refused = preflight({ "extra-large": { ok: false, message: REFUSAL } });
    assert.equal(definition.normalize(selected, context(refused)).tilingSize, AUTO_TILE);
    assert.equal(definition.normalize(selected, context(preflight({}))).tilingSize, "extra-large");
  });

  it("sends a tile size only when it is not Auto", () => {
    const standard = definition.toCreateBody(withVideo, context());
    assert.equal("tilingSize" in standard.params, false);
    const large = definition.toCreateBody({ ...withVideo, tilingSize: "large" }, context());
    assert.equal(large.params.tilingSize, "large");
  });
});
