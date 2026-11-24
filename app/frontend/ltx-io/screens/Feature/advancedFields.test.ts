import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { getIcLoraRecipe } from "../../../lib/ic-lora-recipes.ts";
import { formWithAdvancedFields, valuesForRun } from "./advancedFields.ts";
import {
  createIcLoraRecipeDefinition,
  type IcLoraRecipeContext,
} from "./definitions/icLoraRecipe.ts";
import { specFixture } from "./definitions/specFixture.ts";

const context: IcLoraRecipeContext = {
  specs: specFixture(),
  seed: null,
  videoDurationSeconds: 4,
  videoDurationPending: false,
  videoWidth: 1920,
  videoHeight: 1080,
  videoFps: null,
  tilingAdaptive: true,
};

const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("refine-details"));
const chosen = {
  ...definition.defaults,
  video: { assetId: "clip" },
  tilingOverlap: 0.15,
  carryFrames: 33,
  tilingSize: "large" as const,
};
const form = definition.form(chosen, context);

describe("advanced fields start on Auto", () => {
  const fields = definition.form(definition.defaults, context).advancedFields ?? [];

  it("lists the four controls, each an options field that starts on its first option, Auto", () => {
    assert.deepEqual(
      fields.map((field) => field.id),
      ["tilingOverlap", "carryFrames", "maxWindowFrames", "tilingSize"],
    );
    for (const field of fields) {
      assert.equal(field.kind, "options", field.id);
      assert.equal(field.defaultValue, field.options[0]?.value, field.id);
      assert.match(String(field.options[0]?.label), /^Auto/, field.id);
    }
  });

  it("starts with no override stored and sends none", () => {
    for (const key of ["tilingOverlap", "carryFrames", "maxWindowFrames"] as const) {
      assert.equal(definition.defaults[key], null, key);
    }
    assert.equal(definition.defaults.tilingSize, "auto");
    const body = definition.toCreateBody({ ...definition.defaults, video: { assetId: "clip" } }, context);
    for (const key of ["tilingOverlap", "carryFrames", "maxWindowFrames", "tilingSize"]) {
      assert.equal(key in body.params, false, key);
    }
  });

  it("offers the overlap presets after Auto", () => {
    const overlap = fields.find((field) => field.id === "tilingOverlap");
    assert.deepEqual(
      overlap?.options.map((option) => option.label),
      ["Auto", "0.50", "0.35", "0.25", "0.15", "0.00"],
    );
  });
});

describe("advanced fields gate", () => {
  it("removes the advanced fields while the setting is off and keeps the form object when on", () => {
    assert.ok((form.advancedFields ?? []).length > 0);
    assert.deepEqual(formWithAdvancedFields(form, false).advancedFields, []);
    assert.equal(formWithAdvancedFields(form, true), form);
    const plain = { fields: form.fields };
    assert.equal(formWithAdvancedFields(plain, false), plain);
  });

  it("resets every advanced field to its default for the run while the setting is off", () => {
    const off = valuesForRun(form, chosen, definition.defaults, false);
    assert.equal(off.tilingOverlap, definition.defaults.tilingOverlap);
    assert.equal(off.carryFrames, definition.defaults.carryFrames);
    assert.equal(off.tilingSize, definition.defaults.tilingSize);
    assert.equal(off.video, chosen.video);
    assert.equal(valuesForRun(form, chosen, definition.defaults, true), chosen);
  });

  it("sends no tiling override for a stored advanced choice while the setting is off", () => {
    const off = definition.toCreateBody(
      valuesForRun(form, chosen, definition.defaults, false),
      context,
    );
    for (const key of ["tilingOverlap", "carryFrames", "tilingSize"]) {
      assert.equal(key in off.params, false, key);
    }
    const on = definition.toCreateBody(chosen, context);
    assert.equal(on.params.tilingOverlap, 0.15);
    assert.equal(on.params.carryFrames, 33);
    assert.equal(on.params.tilingSize, "large");
  });
});
