import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { components } from "../../../../generated/backend-openapi.ts";
import { getIcLoraRecipe } from "../../../../lib/ic-lora-recipes.ts";
import { createIcLoraRecipeDefinition, type IcLoraRecipeContext } from "./icLoraRecipe.ts";
import { carryOptions, overlapOptions, windowFrameOptions } from "./icLoraTileOptions.ts";
import { AUTO_NUMBER } from "./icLoraTilingControls.ts";
import { specFixture } from "./specFixture.ts";

type TilePreflight = components["schemas"]["IcLoraTilePreflightResponse"];
type WindowCell = components["schemas"]["IcLoraMaxWindowPreflightCell"];
type Defaults = TilePreflight["defaults"];

const REFUSAL = "This job needs more VRAM than this machine has.";
const WINDOWS = [97, 73, 57, 41];

function preflight(
  cells: Partial<Record<number, Partial<WindowCell>>>,
  defaults: Partial<Defaults> = {},
): TilePreflight {
  return {
    budgetMode: "cautious",
    cells: [],
    defaults: { carryFrames: 25, auto: { kind: "windowed", frames: 97 }, ...defaults },
    maxWindowCells: WINDOWS.map((maxWindowFrames) => ({ maxWindowFrames, ok: true, ...cells[maxWindowFrames] })),
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
const reasons = (options: { disabledReason?: string }[]) =>
  options.map((option) => option.disabledReason != null);

describe("refine details maximum chunk length", () => {
  it("offers Auto first, then 97, 73, 57 and 41, all enabled until the preflight answers", () => {
    const options = windowFrameOptions(context());
    assert.deepEqual(
      options.map((option) => option.value),
      [AUTO_NUMBER, 97, 73, 57, 41],
    );
    assert.deepEqual(reasons(options), [false, false, false, false, false]);
    assert.deepEqual(reasons(carryOptions(context())), [false, false, false, false]);
  });

  it("labels Auto with what the backend runs for this clip", () => {
    const label = (defaults: Partial<Defaults>) =>
      windowFrameOptions(context(preflight({}, defaults)))[0]?.label;
    assert.equal(windowFrameOptions(context())[0]?.label, "Auto");
    assert.equal(label({ auto: { kind: "windowed", frames: 97 } }), "Auto (97 frames)");
    assert.equal(label({ auto: { kind: "single-pass" } }), "Auto (single pass)");
    assert.equal(label({ auto: null }), "Auto");
  });

  it("disables a length the preflight refuses and carries the refusal as the reason", () => {
    const options = windowFrameOptions(context(preflight({ 97: { ok: false, message: REFUSAL } })));
    assert.equal(options.find((option) => option.value === 97)?.disabledReason, REFUSAL);
    assert.equal(options.find((option) => option.value === 73)?.disabledReason, undefined);
    assert.equal(options[0]?.disabledReason, undefined, "Auto is never disabled");
  });

  it("disables a carry the preflight refuses and carries the refusal as the reason", () => {
    const verdict = context({
      ...preflight({}),
      carryCells: [
        { carryFrames: 17, ok: true },
        { carryFrames: 25, ok: true },
        { carryFrames: 33, ok: false, message: REFUSAL },
      ],
    });
    assert.deepEqual(
      carryOptions(verdict).map((option) => [option.value, option.disabledReason ?? null]),
      [[AUTO_NUMBER, null], [17, null], [25, null], [33, REFUSAL]],
    );
  });

  it("labels the Auto carry with what the backend runs for this clip and chunk length", () => {
    assert.equal(carryOptions(context())[0]?.label, "Auto");
    const label = (carryFrames: number) =>
      carryOptions(context(preflight({}, { carryFrames })))[0]?.label;
    assert.equal(label(17), "Auto (17 frames)");
    assert.equal(label(25), "Auto (25 frames)");
  });

  it("labels the Auto overlap with the overlap of the catalog entry, and never disables an overlap", () => {
    assert.equal(overlapOptions(context())[0]?.label, "Auto");
    const options = overlapOptions(context(preflight({}, { overlapFraction: 0.25 })));
    assert.equal(options[0]?.label, "Auto (0.25)");
    assert.deepEqual(reasons(options), options.map(() => false));
  });

  it("keeps a short length when the resolution changes and the preflight has not refused it", () => {
    const high = { ...withVideo, maxWindowFrames: 41, resolution: "1440p" as const };
    const cells: IcLoraRecipeContext = { ...context(), highResCells: ["1440p", "2160p"] };
    assert.equal(definition.normalize(high, cells).maxWindowFrames, 41);
    assert.equal(definition.normalize({ ...high, resolution: "1080p" }, cells).maxWindowFrames, 41);
  });

  it("resets a length the budget refuses to Auto and keeps one it allows", () => {
    const selected = { ...withVideo, maxWindowFrames: 97 };
    const refused = preflight({ 97: { ok: false, message: REFUSAL } });
    assert.equal(definition.normalize(selected, context(refused)).maxWindowFrames, null);
    assert.equal(definition.normalize(selected, context(preflight({}))).maxWindowFrames, 97);
  });

  it("stores Auto as null, and a value that is not offered falls back to Auto", () => {
    const change = {
      kind: "options" as const,
      fieldId: "maxWindowFrames",
      dataKey: "maxWindowFrames" as const,
      value: AUTO_NUMBER,
    };
    assert.equal(definition.applyChange({ ...withVideo, maxWindowFrames: 57 }, change).maxWindowFrames, null);
    assert.equal(definition.normalize({ ...withVideo, maxWindowFrames: 65 }, context()).maxWindowFrames, null);
  });

  it("sends a length as the maximum and sends nothing for Auto", () => {
    const auto = definition.toCreateBody({ ...withVideo, maxWindowFrames: null }, context());
    assert.equal("maxWindowFrames" in auto.params, false);
    const chosen = definition.toCreateBody({ ...withVideo, maxWindowFrames: 57 }, context());
    assert.equal(chosen.params.maxWindowFrames, 57);
    const longest = definition.toCreateBody({ ...withVideo, maxWindowFrames: 97 }, context());
    assert.equal(longest.params.maxWindowFrames, 97);
  });

  it("stores Auto as null for the carry and the overlap", () => {
    const stored = { ...withVideo, carryFrames: 33, tilingOverlap: 0.15 };
    const auto = (dataKey: "carryFrames" | "tilingOverlap") =>
      definition.applyChange(stored, {
        kind: "options",
        fieldId: dataKey,
        dataKey,
        value: AUTO_NUMBER,
      });
    assert.equal(auto("carryFrames").carryFrames, null);
    assert.equal(auto("carryFrames").tilingOverlap, 0.15);
    assert.equal(auto("tilingOverlap").tilingOverlap, null);
  });

  it("sends the carry the user chose, without comparing it to the preflight", () => {
    const body = (carryFrames: number | null, tilePreflight?: TilePreflight) =>
      definition.toCreateBody({ ...withVideo, carryFrames }, context(tilePreflight)).params;
    assert.equal("carryFrames" in body(null), false);
    for (const tilePreflight of [undefined, preflight({})]) {
      assert.equal(body(25, tilePreflight).carryFrames, 25);
      assert.equal(body(17, tilePreflight).carryFrames, 17);
    }
  });

  it("hides the Advanced section until the catalog says the window is adaptive", () => {
    const hidden = { ...context(), tilingAdaptive: undefined };
    assert.equal((definition.form(withVideo, hidden).advancedFields ?? []).length, 0);
    assert.ok((definition.form(withVideo, context()).advancedFields ?? []).length > 0);
  });
});
