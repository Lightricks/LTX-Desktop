import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { getLoraRecipe } from "../../../../lib/lora-recipes.ts";
import {
  resolveFeatureFormValues,
  resolveFeatureSeedValues,
  type FeatureFieldChange,
} from "../types.ts";
import { specFixtureBothOfferings } from "./specFixture.ts";
import { createLoraRecipeDefinition } from "./loraRecipe.ts";
import type { LoraRecipeSchema, LoraRecipeValues } from "./loraRecipe.ts";

const recipe = getLoraRecipe("cozy-felt");
const definition = createLoraRecipeDefinition(recipe);

describe("createLoraRecipeDefinition", () => {
  it("builds a create body of catalogId + scale + offering id, never a ref", () => {
    const body = definition.toCreateBody(
      { ...definition.defaults, prompt: "a felt fox", strength: 1.5 },
      { specs: undefined },
    );

    assert.equal(body.contract_version, 1);
    assert.equal(body.params.prompt, "a felt fox");
    assert.equal(body.params.model, "ltx-2.5-fast");
    assert.equal(body.params.catalogId, "cozy-felt-style");
    assert.equal(body.params.scale, 1.5);
    assert.equal(body.params.cameraMotion, "none");
    // The io recipe FE never sends a filesystem ref; the backend resolves it.
    assert.ok(!("ref" in body.params));
    assert.ok(!("loras" in body.params));
    assert.equal(body.inputs, undefined);
  });

  it("seeds strength from a stored generation's lora scale", () => {
    const values = definition.fromGeneration(
      {
        params: {
          prompt: "stored scene",
          model: "ltx-2.5-fast",
          resolution: "720p",
          duration: 8,
          fps: 24,
          loras: [{ ref: "x", scale: 2.25, catalogId: "cozy-felt-style" }],
        },
      },
      { specs: specFixtureBothOfferings("LTX 2.3 Fast") },
    );

    assert.ok(values);
    assert.equal(values.prompt, "stored scene");
    assert.equal(values.model, "ltx-2.5-fast");
    assert.equal(values.strength, 2.25);
  });

  it("defaults the prompt to the recipe seed", () => {
    assert.equal(definition.defaults.prompt, recipe.seedPrompt);
    assert.equal(definition.defaults.model, "ltx-2.5-fast");
    assert.ok(definition.defaults.prompt.trim().length > 0);
  });

  it("requires a prompt", () => {
    const issues = definition.validate(
      { ...definition.defaults, prompt: "   " },
      { specs: undefined },
    );
    assert.equal(issues.length, 1);
    assert.equal(issues[0].fieldId, "prompt");
  });

  it("warns on a model the style LoRA was not trained for", () => {
    const form = definition.form(definition.defaults, {
      specs: specFixtureBothOfferings("LTX 2.5 Fast"),
      catalogVariants: [
        {
          id: "default",
          label: "LTX-2.3",
          baseModel: "LTX-2.3",
          downloaded: true,
        },
      ],
    });
    assert.equal(form.fields.some((field) => field.dataKey === "variant"), false);
    const model = form.fields.find((field) => field.dataKey === "model");
    assert.ok(model && model.kind === "options");
    if (model.kind !== "options") return;
    assert.equal(
      model.options.find((option) => option.value === "ltx-2.5-fast")?.warning,
      "You only have the LTX-2.3 Variant. Results with LTX-2.5 may not be optimal.",
    );
    assert.equal(
      model.options.find((option) => option.value === "ltx-2.3-fast")?.warning,
      undefined,
    );
  });
});

describe("LoRA supported models", () => {
  const specs = specFixtureBothOfferings("LTX 2.5 Fast");

  it("disables an installed model the catalog entry does not support", () => {
    const form = definition.form(definition.defaults, {
      specs,
      supportedModels: ["LTX-2.5"],
    });
    const model = form.fields.find((field) => field.dataKey === "model");
    assert.ok(model && model.kind === "options");
    if (model.kind !== "options") return;
    assert.match(
      model.options.find((option) => option.value === "ltx-2.3-fast")?.disabledReason ?? "",
      /LTX-2\.5 only/,
    );
    assert.equal(
      model.options.find((option) => option.value === "ltx-2.5-fast")?.disabledReason,
      undefined,
    );
  });

  it("moves a stored unsupported model to a supported one", () => {
    const stored = { ...definition.defaults, model: "ltx-2.3-fast" as const };
    assert.equal(
      definition.normalize(stored, { specs, supportedModels: ["LTX-2.5"] }).model,
      "ltx-2.5-fast",
    );
    assert.equal(definition.normalize(stored, { specs }).model, "ltx-2.3-fast");
  });

  it("blocks Generate on a model the catalog entry does not support", () => {
    const values = {
      ...definition.defaults,
      prompt: "a prompt",
      model: "ltx-2.3-fast" as const,
    };
    const blocked = definition.validate(values, { specs, supportedModels: ["LTX-2.5"] });
    assert.deepEqual(
      blocked.map((issue) => issue.id),
      ["model-unsupported"],
    );
    assert.deepEqual(definition.validate(values, { specs }), []);
  });
});

describe("LTX-2 weights", () => {
  it("do not warn on either model", () => {
    const form = definition.form(definition.defaults, {
      specs: specFixtureBothOfferings("LTX 2.5 Fast"),
      catalogVariants: [
        { id: "ltx-2__dolly-in", label: "Default", baseModel: "LTX-2", downloaded: true },
      ],
    });
    const model = form.fields.find((field) => field.dataKey === "model");
    assert.ok(model && model.kind === "options");
    if (model.kind !== "options") return;
    assert.deepEqual(
      model.options.map((option) => option.warning),
      model.options.map(() => undefined),
    );
  });
});

describe("lora recipe offering default", () => {
  it("seeds the offering that matches Settings' active local row", () => {
    const seeded = resolveFeatureSeedValues(definition, {
      context: { specs: specFixtureBothOfferings("LTX 2.3 Fast") },
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: undefined,
    });
    assert.equal(seeded.model, "ltx-2.3-fast");
  });

  it("restores last-generation prompt but maps unsupported model to the Settings offering", () => {
    const seeded = resolveFeatureSeedValues(definition, {
      context: { specs: specFixtureBothOfferings("LTX 2.3 Fast") },
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: {
        params: {
          prompt: "a felt fox",
          model: "fast",
          aspectRatio: "9:16",
          resolution: "720p",
          duration: 8,
          fps: 24,
          loras: [{ ref: "x", scale: 1.5, catalogId: "cozy-felt-style" }],
        },
      },
    });
    assert.equal(seeded.model, "ltx-2.3-fast");
    assert.equal(seeded.prompt, "a felt fox");
    assert.equal(seeded.strength, 1.5);
  });

  it("keeps a last-generation offering id even when Settings is a different row", () => {
    const seeded = resolveFeatureSeedValues(definition, {
      context: { specs: specFixtureBothOfferings("LTX 2.3 Fast") },
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: {
        params: {
          prompt: "keep this prompt",
          model: "ltx-2.5-fast",
          aspectRatio: "16:9",
          resolution: "720p",
          duration: 8,
          fps: 24,
        },
      },
    });
    assert.equal(seeded.model, "ltx-2.5-fast");
  });

  it("shows the Settings-matching offering on first paint before the store write", () => {
    const displayed = resolveFeatureFormValues(definition, {
      context: { specs: specFixtureBothOfferings("LTX 2.3 Fast") },
      contextReady: true,
      storedValues: undefined,
      generationsFetched: true,
      lastGenerationSpec: undefined,
    });
    assert.equal(displayed.model, "ltx-2.3-fast");
  });

  it("resetValues picks the Settings-matching offering, not hardcoded 2.5", () => {
    const reset = definition.resetValues?.({
      specs: specFixtureBothOfferings("LTX 2.3 Fast"),
    });
    assert.equal(reset?.model, "ltx-2.3-fast");
    assert.equal(reset?.prompt, recipe.seedPrompt);
  });

  it("applyChange falls back to the Settings-matching offering, not hardcoded 2.5", () => {
    const change: FeatureFieldChange<LoraRecipeSchema> = {
      kind: "options",
      fieldId: "aspectRatio",
      dataKey: "aspectRatio",
      value: "9:16",
    };

    const next = definition.applyChange(
      {
        ...definition.defaults,
        model: "fast" as LoraRecipeValues["model"],
      },
      change,
      { specs: specFixtureBothOfferings("LTX 2.3 Fast") },
    );

    assert.equal(next.model, "ltx-2.3-fast");
  });
});

describe("dolly-in i2v recipe definition", () => {
  const dollyIn = getLoraRecipe("dolly-in");
  const definition = createLoraRecipeDefinition(dollyIn);
  const seed = { assetId: "dolly-in-seed" };

  it("puts Start Frame above the prompt and defaults aspect to Auto", () => {
    const form = definition.form(
      { ...definition.defaults, startFrame: seed },
      { specs: undefined, seed },
    );
    assert.equal(form.fields[0]?.kind, "image-asset");
    assert.equal(form.fields[0] && "dataKey" in form.fields[0] ? form.fields[0].dataKey : null, "startFrame");
    assert.equal(definition.defaults.aspectRatio, "auto");
    assert.equal(definition.defaults.startFrame, null);
  });

  it("does not show Start Frame on Cozy Felt", () => {
    const cozy = createLoraRecipeDefinition(recipe);
    const cozyForm = cozy.form(cozy.defaults, { specs: undefined });
    assert.equal(
      cozyForm.fields.some((field) => field.kind === "image-asset"),
      false,
    );
  });

  it("seeds Start Frame from the packaged still and sends it on the create body", () => {
    const seeded = definition.initialValues?.({ specs: undefined, seed });
    assert.deepEqual(seeded?.startFrame, seed);
    const body = definition.toCreateBody(
      { ...definition.defaults, startFrame: seed, prompt: "push in on her face" },
      { specs: undefined, seed },
    );
    assert.equal(body.inputs?.startFrame.assetId, "dolly-in-seed");
    assert.equal(body.params.catalogId, "dolly-in");
    assert.equal(body.params.aspectRatio, "auto");
    assert.ok(!("ref" in body.params));
  });

  it("requires a start frame", () => {
    const issues = definition.validate(
      { ...definition.defaults, prompt: "push in" },
      { specs: undefined },
    );
    assert.equal(issues.some((issue) => issue.fieldId === "startFrame"), true);
  });

  it("restores Start Frame from a stored generation", () => {
    const values = definition.fromGeneration(
      {
        params: {
          prompt: "stored dolly",
          model: "ltx-2.5-fast",
          aspectRatio: "auto",
          resolution: "720p",
          duration: 8,
          fps: 24,
          loras: [{ ref: "", scale: 1.25, catalogId: "dolly-in" }],
        },
        inputs: { startFrame: { assetId: "history-still" } },
      },
      { specs: specFixtureBothOfferings("LTX 2.3 Fast"), seed },
    );
    assert.ok(values);
    assert.equal(values.startFrame?.assetId, "history-still");
    assert.equal(values.strength, 1.25);
    assert.equal(values.aspectRatio, "auto");
  });
});

describe("transition i2v recipe definition", () => {
  const transition = getLoraRecipe("transition");
  const definition = createLoraRecipeDefinition(transition);
  const start = { assetId: "transition-start" };
  const end = { assetId: "transition-end" };

  it("shows Start Frame and End Frame above the prompt", () => {
    const form = definition.form(
      { ...definition.defaults, startFrame: start, endFrame: end },
      { specs: undefined, seed: start, endSeed: end },
    );
    const imageFields = form.fields.filter((field) => field.kind === "image-asset");
    assert.deepEqual(
      imageFields.map((field) => ("dataKey" in field ? field.dataKey : null)),
      ["startFrame", "endFrame"],
    );
  });

  it("sends start and end frames on the create body", () => {
    const body = definition.toCreateBody(
      {
        ...definition.defaults,
        startFrame: start,
        endFrame: end,
        prompt: "a wave becomes a mountain",
      },
      { specs: undefined, seed: start, endSeed: end },
    );
    assert.equal(body.inputs?.startFrame.assetId, "transition-start");
    assert.equal(body.inputs?.endFrame?.assetId, "transition-end");
    assert.equal(body.params.catalogId, "transition");
  });

  it("requires an end frame", () => {
    const issues = definition.validate(
      { ...definition.defaults, prompt: "morph", startFrame: start },
      { specs: undefined },
    );
    assert.equal(issues.some((issue) => issue.fieldId === "endFrame"), true);
  });
});
