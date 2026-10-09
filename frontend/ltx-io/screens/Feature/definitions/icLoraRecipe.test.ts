import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  getIcLoraRecipe,
  IC_LORA_RECIPES,
  type IcLoraRecipe,
} from "../../../../lib/ic-lora-recipes.ts";
import type { CatalogVariantChoice } from "../../../lib/catalogVariantChoice.ts";
import { planIcLoraRecipeSeed } from "../../../hooks/icLoraRecipeSeed.ts";
import { specFixture, specFixtureBothOfferings } from "./specFixture.ts";
import {
  createIcLoraRecipeDefinition,
  IC_LORA_RECIPE_LORA_STRENGTH,
  type IcLoraRecipeContext,
} from "./icLoraRecipe.ts";

const CAPS = {
  t2v: true,
  i2v: true,
  a2v: true,
  ic_lora: true,
  retake: true,
  extend: true,
  multi_keyframe: true,
  multi_keyframe_max_count: 3,
  user_loras: true,
  camera_motion: true,
  auto_duration: false,
} as const;

function context(): IcLoraRecipeContext {
  const specs = specFixture();
  return {
    specs: {
      ...specs,
      downloaded_local_models: specs.downloaded_local_models.map((item) => ({
        ...item,
        spec: { ...item.spec, capabilities: { ...CAPS } },
      })),
    },
    seed: null,
    videoDurationSeconds: 4,
    videoDurationPending: false,
    videoWidth: 320,
    videoHeight: 240,
    videoFps: null,
  };
}

describe("ic-lora recipe form", () => {
  it("offers only resolutions inside the 540p envelope and locks aspect", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("day-to-night"));
    const form = definition.form(
      { ...definition.defaults, video: { assetId: "clip" } },
      context(),
    );
    const dataKeys = form.fields.map((field) => field.dataKey);
    assert.equal(dataKeys.includes("aspectRatio"), false);
    const video = form.fields.find((field) => field.dataKey === "video");
    assert.equal(video?.label, "Reference Video");
    assert.equal(video?.kind === "video-asset" && video.maxDurationSeconds, 10);
    assert.equal(video?.label.includes("16:9"), false);
    const audio = form.fields.find((field) => field.dataKey === "audioMode");
    assert.ok(audio && audio.kind === "options");
    if (audio?.kind !== "options") return;
    assert.deepEqual(
      audio.options.map((option) => option.label),
      ["Off", "Source", "Generated"],
    );
    assert.equal(audio.forcePicker, undefined);
    const resolution = form.fields.find((field) => field.dataKey === "resolution");
    assert.ok(resolution && resolution.kind === "options");
    const values = resolution.options.map((option) => String(option.value));
    assert.deepEqual(values, ["270p", "360p", "540p"]);
    const strength = form.fields.find((field) => field.dataKey === "loraStrength");
    assert.ok(strength && strength.kind === "slider");
    if (strength?.kind !== "slider") return;
    assert.equal(strength.label, "Style strength");
    assert.equal(strength.min, 0);
    assert.equal(strength.max, 2);
    assert.equal(definition.defaults.loraStrength, IC_LORA_RECIPE_LORA_STRENGTH);
    assert.equal(definition.defaults.audioMode, "source");
    const fromCatalog = createIcLoraRecipeDefinition(getIcLoraRecipe("day-to-night"), {
      lora_strength: 1.5,
      audio_mode: "generated",
    });
    assert.equal(fromCatalog.defaults.loraStrength, 1.5);
    assert.equal(fromCatalog.defaults.audioMode, "generated");
  });

  it("accepts a blank prompt only when the catalog entry allows it", () => {
    const blank = { video: { assetId: "clip" }, prompt: "  " };
    const required = createIcLoraRecipeDefinition(getIcLoraRecipe("day-to-night"));
    const requiredIssues = required.validate(
      { ...required.defaults, ...blank },
      context(),
    );
    assert.deepEqual(
      requiredIssues.map((issue) => issue.id),
      ["prompt-required"],
    );

    const optional = createIcLoraRecipeDefinition(getIcLoraRecipe("alpha-gen"), {
      allows_empty_prompt: true,
    });
    assert.equal(optional.defaults.prompt, "");
    assert.deepEqual(optional.validate({ ...optional.defaults, ...blank }, context()), []);
    assert.equal(
      optional.toCreateBody({ ...optional.defaults, ...blank }, context()).params.prompt,
      "",
    );
  });

  it("hides prompt, audio and style strength for AlphaGen and sends the catalog defaults", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("alpha-gen"), {
      lora_strength: 1,
      audio_mode: "source",
    });
    const dataKeys = definition.form(
      { ...definition.defaults, video: { assetId: "clip" } },
      context(),
    ).fields.map((field) => field.dataKey);
    for (const hidden of ["prompt", "audioMode", "loraStrength"]) {
      assert.equal(dataKeys.includes(hidden as never), false, hidden);
    }
    for (const shown of ["video", "resolution", "model"]) {
      assert.equal(dataKeys.includes(shown as never), true, shown);
    }

    // A stale stored value must not reach the request, and a blank prompt stays valid.
    const stale = {
      ...definition.defaults,
      video: { assetId: "clip" },
      prompt: "old prompt",
      audioMode: "off" as const,
      loraStrength: 0.4,
    };
    assert.deepEqual(definition.validate(stale, context()), []);
    const params = definition.toCreateBody(stale, context()).params;
    assert.equal(params.prompt, "");
    assert.equal(params.audioMode, "source");
    assert.equal(params.scale, 1);
  });

  it("keeps every control for a recipe that does not hide any", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("day-to-night"));
    const dataKeys = definition.form(
      { ...definition.defaults, video: { assetId: "clip" } },
      context(),
    ).fields.map((field) => field.dataKey);
    for (const shown of ["prompt", "audioMode", "loraStrength"]) {
      assert.equal(dataKeys.includes(shown as never), true, shown);
    }
  });

  it("disables an installed model the catalog entry does not support", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("alpha-gen"));
    const values = { ...definition.defaults, video: { assetId: "clip" } };
    const modelOptions = (supportedModels?: readonly string[]) => {
      const field = definition
        .form(values, { ...bothOfferings(), supportedModels })
        .fields.find((candidate) => candidate.dataKey === "model");
      assert.ok(field && field.kind === "options");
      return field.kind === "options" ? field.options : [];
    };

    const only25 = modelOptions(["LTX-2.5"]);
    assert.equal(only25.length, 2);
    for (const option of only25) {
      const supported = String(option.value).startsWith("ltx-2.5");
      assert.equal(option.disabledReason == null, supported, String(option.value));
    }
    assert.match(
      only25.find((option) => String(option.value).startsWith("ltx-2.3"))?.disabledReason ?? "",
      /LTX-2\.5 only/,
    );
    // Before the catalog loads, and for an entry that lists both, nothing is disabled.
    for (const supported of [undefined, ["LTX-2.3", "LTX-2.5"]]) {
      assert.equal(modelOptions(supported).every((option) => option.disabledReason == null), true);
    }
  });

  it("moves a stored unsupported model to a supported one", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("alpha-gen"));
    const stored = { ...definition.defaults, model: "ltx-2.3-fast" as const };
    const context25 = { ...bothOfferings(), supportedModels: ["LTX-2.5"] };
    assert.equal(definition.normalize(stored, context25).model, "ltx-2.5-fast");
    assert.equal(
      definition.normalize(stored, { ...bothOfferings(), supportedModels: undefined }).model,
      "ltx-2.3-fast",
    );
  });

  it("blocks Generate when only an unsupported model is available", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("alpha-gen"));
    const values = {
      ...definition.defaults,
      video: { assetId: "clip" },
      model: "ltx-2.3-fast" as const,
    };
    const only23 = {
      ...context(),
      supportedModels: ["LTX-2.5"],
    };
    const issues = definition.validate(values, only23);
    assert.deepEqual(
      issues.map((issue) => issue.id),
      ["model-unsupported"],
    );
    assert.equal(issues[0]?.fieldId, "model");
    assert.deepEqual(definition.validate(values, { ...only23, supportedModels: undefined }), []);
  });

  it("offers Original and slower rates, and the trim cap follows the selection", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("day-to-night"));
    const phone = { ...context(), videoFps: 30 };
    const atOriginal = definition.form(
      { ...definition.defaults, video: { assetId: "clip" }, fps: 25 },
      phone,
    );
    const fps = atOriginal.fields.find((field) => field.dataKey === "fps");
    assert.ok(fps && fps.kind === "options");
    if (fps?.kind !== "options") return;
    assert.deepEqual(
      fps.options.map((option) => option.label),
      ["24", "25 (Original)"],
    );
    const video = atOriginal.fields.find((field) => field.dataKey === "video");
    assert.equal(video?.kind === "video-asset" && video.maxDurationSeconds, 9.6);
    const at24 = definition.form(
      { ...definition.defaults, video: { assetId: "clip" }, fps: 24 },
      phone,
    );
    const longer = at24.fields.find((field) => field.dataKey === "video");
    assert.equal(longer?.kind === "video-asset" && longer.maxDurationSeconds, 10);
    assert.equal(definition.defaults.fps, null);
    const atDefault = definition.form(
      { ...definition.defaults, video: { assetId: "clip" } },
      phone,
    );
    const defaultVideo = atDefault.fields.find((field) => field.dataKey === "video");
    assert.equal(defaultVideo?.kind === "video-asset" && defaultVideo.maxDurationSeconds, 9.6);
    const clip = { ...definition.defaults, video: { assetId: "clip" } };
    assert.equal(definition.toCreateBody(clip).params.fps, undefined);
    assert.equal(definition.toCreateBody({ ...clip, fps: 24 }).params.fps, 24);
    assert.equal(definition.normalize({ ...clip, fps: 25 }, phone).fps, null);
    assert.equal(definition.normalize({ ...clip, fps: 24 }, phone).fps, 24);
    const onlyOriginal = definition.form(
      { ...definition.defaults, video: { assetId: "clip" } },
      { ...context(), videoFps: 24 },
    );
    assert.equal(
      onlyOriginal.fields.some((field) => field.dataKey === "fps"),
      false,
    );
  });

  it("shows the Original rate in the fps control while the stored rate is unset", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("day-to-night"));
    const defaultDefaults = (videoFps: number) => {
      const form = definition.form(
        { ...definition.defaults, video: { assetId: "clip" } },
        { ...context(), videoFps },
      );
      const fps = form.fields.find((field) => field.dataKey === "fps");
      return fps?.kind === "options" ? fps.defaultValue : undefined;
    };
    assert.equal(defaultDefaults(25), 25);
    assert.equal(defaultDefaults(30), 25);
    assert.equal(defaultDefaults(60), 50);
  });

  it("offers 720p and 1080p only when the source is at least that large", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("day-to-night"));
    const clip = { ...definition.defaults, video: { assetId: "clip" } };
    const offered = (videoWidth: number | null, videoHeight: number | null) => {
      const form = definition.form(clip, { ...context(), videoWidth, videoHeight });
      const field = form.fields.find((item) => item.dataKey === "resolution");
      return field?.kind === "options" ? field.options.map((option) => option.value) : [];
    };
    assert.deepEqual(offered(1920, 1080), ["270p", "360p", "540p", "720p", "1080p"]);
    assert.deepEqual(offered(1280, 720), ["270p", "360p", "540p", "720p"]);
    assert.deepEqual(offered(960, 540), ["270p", "360p", "540p"]);
    assert.deepEqual(offered(null, null), ["270p", "360p", "540p", "720p", "1080p"]);
    const picked = { ...clip, resolution: "720p" as const };
    assert.equal(
      definition.normalize(picked, { ...context(), videoWidth: 960, videoHeight: 540 })
        .resolution,
      "540p",
    );
    assert.equal(
      definition.normalize(picked, { ...context(), videoWidth: 1280, videoHeight: 720 })
        .resolution,
      "720p",
    );
  });

  it("tags the resolution closest to the source, and not above it, as Original", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("day-to-night"));
    const clip = { ...definition.defaults, video: { assetId: "clip" } };
    const labels = (videoWidth: number | null, videoHeight: number | null) => {
      const form = definition.form(clip, { ...context(), videoWidth, videoHeight });
      const field = form.fields.find((item) => item.dataKey === "resolution");
      return field?.kind === "options" ? field.options.map((option) => option.label) : [];
    };
    assert.deepEqual(labels(1920, 1080), [
      "270p",
      "360p",
      "540p",
      "720p",
      "1080p (Original)",
    ]);
    assert.deepEqual(labels(1280, 720), ["270p", "360p", "540p", "720p (Original)"]);
    assert.deepEqual(labels(640, 360), ["270p", "360p (Original)", "540p"]);
    assert.deepEqual(labels(null, null), ["270p", "360p", "540p", "720p", "1080p"]);
  });

  it("warns about slow runs at 720p and higher on a low performance machine only", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("day-to-night"));
    const clip = { ...definition.defaults, video: { assetId: "clip" } };
    const warned = (lowPerformanceMachine: boolean | undefined) => {
      const base = context();
      const form = definition.form(clip, {
        ...base,
        specs: base.specs && { ...base.specs, low_performance_machine: lowPerformanceMachine },
        videoWidth: 1920,
        videoHeight: 1080,
      });
      const field = form.fields.find((item) => item.dataKey === "resolution");
      return field?.kind === "options"
        ? field.options.filter((option) => option.warning != null).map((option) => option.value)
        : [];
    };
    assert.deepEqual(warned(true), ["720p", "1080p"]);
    assert.deepEqual(warned(false), []);
    assert.deepEqual(warned(undefined), []);
  });

  it("defaults to 720p and falls back to the largest offered cell for a smaller source", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("day-to-night"));
    assert.equal(definition.defaults.resolution, "720p");
    const clip = { ...definition.defaults, video: { assetId: "clip" } };
    const settle = (videoWidth: number | null, videoHeight: number | null) =>
      definition.normalize(clip, { ...context(), videoWidth, videoHeight }).resolution;
    assert.equal(settle(1920, 1080), "720p");
    assert.equal(settle(null, null), "720p");
    assert.equal(settle(960, 540), "540p");
    assert.equal(settle(320, 240), "540p");
  });

  it("offers every resolution for a small source only to a recipe that upscales it", () => {
    const resolutions = (recipeId: "day-to-night" | "restore") => {
      const definition = createIcLoraRecipeDefinition(getIcLoraRecipe(recipeId));
      const clip = { ...definition.defaults, video: { assetId: "clip" } };
      const form = definition.form(clip, { ...context(), videoWidth: 480, videoHeight: 270 });
      const field = form.fields.find((item) => item.dataKey === "resolution");
      return field?.kind === "options" ? field.options.map((option) => option.value) : [];
    };
    assert.deepEqual(resolutions("day-to-night"), ["270p", "360p", "540p"]);
    assert.deepEqual(resolutions("restore"), ["270p", "360p", "540p", "720p", "1080p"]);
  });

  it("does not reingest the reference after the form was cleared", () => {
    const recipe = getIcLoraRecipe("day-to-night");
    assert.equal(
      planIcLoraRecipeSeed(recipe, {
        hasStoredValues: true,
        generationsReady: true,
        generationsFailed: false,
        lastGenerationSpec: undefined,
        appliedSeedRevision: recipe.seed.revision,
        packagedSeedRevision: recipe.seed.revision,
      }),
      "skip",
    );
  });

  it("hides Variant until two checkpoints are installed, then lists only those in catalog order", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("day-to-night"));
    const variants: CatalogVariantChoice[] = [
      { id: "ltx-2.5__day-to-night", label: "LTX-2.5", baseModel: "LTX-2.5", downloaded: true },
      { id: "default", label: "LTX-2.3", baseModel: "LTX-2.3", downloaded: false },
    ];
    const one = definition.form(definition.defaults, {
      ...context(),
      catalogVariants: variants,
    });
    assert.equal(one.fields.some((field) => field.dataKey === "variant"), false);

    const both = definition.form(definition.defaults, {
      ...context(),
      catalogVariants: variants.map((variant) => ({ ...variant, downloaded: true })),
    });
    const variant = both.fields.find((field) => field.dataKey === "variant");
    const model = both.fields.find((field) => field.dataKey === "model");
    assert.ok(variant && variant.kind === "options");
    assert.ok(model);
    assert.equal(both.fields.indexOf(variant), both.fields.indexOf(model) + 1);
    if (variant.kind !== "options") return;
    assert.deepEqual(
      variant.options.map((option) => option.value),
      ["ltx-2.5__day-to-night", "default"],
    );
    assert.equal(
      variant.options[1]?.warning,
      "This IC-LoRA Variant was trained on LTX-2.3. Results with LTX-2.5 may not be optimal.",
    );
    assert.equal(variant.options[0]?.warning, undefined);
  });

  it("does not follow the selected model when both checkpoints are installed", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("day-to-night"));
    const catalogVariants: CatalogVariantChoice[] = [
      { id: "ltx-2.5__day-to-night", label: "LTX-2.5", baseModel: "LTX-2.5", downloaded: true },
      { id: "default", label: "LTX-2.3", baseModel: "LTX-2.3", downloaded: true },
    ];
    const ctx = { ...bothOfferings(), catalogVariants };
    const settled = definition.normalize(
      { ...definition.defaults, model: "ltx-2.5-fast", variant: "default" },
      ctx,
    );
    assert.equal(settled.variant, "ltx-2.5__day-to-night");
    const switched = definition.normalize({ ...settled, model: "ltx-2.3-fast" }, ctx);
    assert.equal(switched.variant, "default");
    const sent = definition.toCreateBody(
      { ...switched, video: { assetId: "clip" } },
      ctx,
    );
    assert.equal(sent.params.variantId, "default");
    const form = definition.form(switched, ctx);
    assert.equal(form.fields.some((field) => field.dataKey === "variant"), false);
    const model = form.fields.find((field) => field.dataKey === "model");
    assert.ok(model && model.kind === "options");
    if (model.kind !== "options") return;
    assert.equal(model.options.every((option) => option.warning == null), true);
  });

  it("stays quiet when two models are installed and only one checkpoint is", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("day-to-night"));
    const form = definition.form(definition.defaults, {
      ...bothOfferings(),
      catalogVariants: [
        { id: "ltx-2.5__day-to-night", label: "LTX-2.5", baseModel: "LTX-2.5", downloaded: false },
        { id: "default", label: "LTX-2.3", baseModel: "LTX-2.3", downloaded: true },
      ],
    });
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

  it("sends the selected installed variant and omits one when nothing is installed", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("day-to-night"));
    const values = {
      ...definition.defaults,
      video: { assetId: "clip" },
      variant: "default",
    };
    const selected = definition.toCreateBody(values, {
      ...context(),
      catalogVariants: [
        { id: "ltx-2.5__day-to-night", label: "LTX-2.5", baseModel: "LTX-2.5", downloaded: true },
        { id: "default", label: "LTX-2.3", baseModel: "LTX-2.3", downloaded: true },
      ],
    });
    assert.equal(selected.params.variantId, "default");

    const none = definition.toCreateBody(values, {
      ...context(),
      catalogVariants: [
        { id: "ltx-2.5__day-to-night", label: "LTX-2.5", baseModel: "LTX-2.5", downloaded: false },
        { id: "default", label: "LTX-2.3", baseModel: "LTX-2.3", downloaded: false },
      ],
    });
    assert.equal(none.params.variantId, undefined);
  });

  it("sends the variant that the download installed when catalogVariants changes between two calls", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("day-to-night"));
    const values = {
      ...definition.defaults,
      video: { assetId: "clip" },
      variant: "default",
    };
    const variants = (downloaded25: boolean): CatalogVariantChoice[] => [
      { id: "ltx-2.5__day-to-night", label: "LTX-2.5", baseModel: "LTX-2.5", downloaded: downloaded25 },
      { id: "default", label: "LTX-2.3", baseModel: "LTX-2.3", downloaded: false },
    ];
    const before = definition.toCreateBody(values, { ...context(), catalogVariants: variants(false) });
    const after = definition.toCreateBody(values, { ...context(), catalogVariants: variants(true) });
    assert.equal(before.params.variantId, undefined);
    assert.equal(after.params.variantId, "ltx-2.5__day-to-night");
  });

  it("keeps a stored variant that is still installed, including before the catalog loads", () => {
    const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("day-to-night"));
    const installed: CatalogVariantChoice[] = [
      { id: "ltx-2.5__day-to-night", label: "LTX-2.5", baseModel: "LTX-2.5", downloaded: true },
      { id: "default", label: "LTX-2.3", baseModel: "LTX-2.3", downloaded: true },
    ];
    const restored = definition.fromGeneration(
      {
        params: {
          prompt: "night street",
          model: "ltx-2.5-fast",
          resolution: "540p",
          audioMode: "generated",
          loras: [{ ref: "", scale: 0.4, variantId: "default" }],
        },
        inputs: { video: { assetId: "clip" } },
      },
      context(),
    );
    assert.ok(restored);
    const held = definition.normalize(restored, { ...context(), catalogVariants: [] });
    assert.equal(held.variant, "default");
    const settled = definition.normalize(held, { ...context(), catalogVariants: installed });
    assert.equal(settled.variant, "default");
    assert.equal(settled.loraStrength, 0.4);
  });
});

function bothOfferings(): IcLoraRecipeContext {
  const specs = specFixtureBothOfferings();
  return {
    ...context(),
    specs: {
      ...specs,
      downloaded_local_models: specs.downloaded_local_models.map((item) => ({
        ...item,
        spec: { ...item.spec, capabilities: { ...CAPS } },
      })),
    },
  };
}

describe("ic-lora recipe reference image", () => {
  const recipes: readonly IcLoraRecipe[] = IC_LORA_RECIPES;

  it("shows an image field if and only if the recipe has a reference image", () => {
    for (const recipe of recipes) {
      const definition = createIcLoraRecipeDefinition(recipe);
      const form = definition.form(
        { ...definition.defaults, video: { assetId: "clip" } },
        context(),
      );
      const hasImageField = form.fields.some((field) => field.dataKey === "image");
      assert.equal(hasImageField, recipe.referenceImage != null, recipe.id);
    }
  });

  it("never sends an image or waits for one when the recipe has no reference image", () => {
    for (const recipe of recipes.filter((candidate) => candidate.referenceImage == null)) {
      const definition = createIcLoraRecipeDefinition(recipe);
      const values = {
        ...definition.defaults,
        video: { assetId: "clip" },
        image: { assetId: "stray-image" },
      };
      assert.equal("image" in definition.toCreateBody(values, context()).inputs, false, recipe.id);
      assert.equal(definition.isReady(values, context()), true, recipe.id);
    }
  });
});

describe("ic-lora recipe Layout to Render", () => {
  const recipe = getIcLoraRecipe("layout-to-render");
  const definition = createIcLoraRecipeDefinition(recipe);
  const clip = { ...definition.defaults, video: { assetId: "clip" } };
  const look = { assetId: "look" };

  it("sends a stored 1080p cell as 720p for a source that only offers 720p", () => {
    const body = definition.toCreateBody(
      { ...clip, resolution: "1080p" },
      { ...context(), videoWidth: 1280, videoHeight: 720 },
    );
    assert.equal(body.params.resolution, "720p");
    const full = definition.toCreateBody(
      { ...clip, resolution: "1080p" },
      { ...context(), videoWidth: 1920, videoHeight: 1080 },
    );
    assert.equal(full.params.resolution, "1080p");
  });

  it("is not ready until the clip width and height are both known", () => {
    const sized = { ...context(), referenceImageRequired: false };
    assert.equal(definition.isReady(clip, sized), true);
    assert.equal(definition.isReady(clip, { ...sized, videoWidth: null }), false);
    assert.equal(definition.isReady(clip, { ...sized, videoHeight: null }), false);
  });

  it("is not ready while the catalog flag reference_image_required is not loaded", () => {
    const withImage = { ...clip, image: look };
    assert.equal(definition.isReady(withImage, context()), false);
    assert.equal(
      definition.isReady(withImage, { ...context(), referenceImageRequired: false }),
      true,
    );
  });

  it("needs the look image to be ready only when the catalog flag is true", () => {
    const ready = (referenceImageRequired: boolean, image: typeof look | null) =>
      definition.isReady({ ...clip, image }, { ...context(), referenceImageRequired });
    assert.equal(ready(true, null), false);
    assert.equal(ready(true, look), true);
    assert.equal(ready(false, null), true);
  });

  it("reports the missing look image only when the catalog flag is true", () => {
    const issues = (referenceImageRequired: boolean | undefined) =>
      definition.validate(clip, { ...context(), referenceImageRequired });
    assert.deepEqual(
      issues(true).map((issue) => [issue.id, issue.fieldId, issue.message]),
      [["image-required", "image", "Look Image is required."]],
    );
    assert.deepEqual(issues(false), []);
    assert.deepEqual(issues(undefined), []);
    assert.deepEqual(
      definition.validate(
        { ...clip, image: look },
        { ...context(), referenceImageRequired: true },
      ),
      [],
    );
  });

  describe("restoring a saved generation", () => {
    const seed = { assetId: "packaged-look" };
    const restore = (inputs: Record<string, unknown>) =>
      definition.fromGeneration(
        {
          params: { prompt: "a train", model: "ltx-2.5-fast", resolution: "540p" },
          inputs: { video: { assetId: "clip" }, ...inputs },
        },
        { ...context(), imageSeed: seed },
      );

    it("keeps a saved empty look image instead of the packaged one", () => {
      assert.equal(restore({ image: null })?.image, null);
    });

    it("uses the packaged look image only when the saved inputs have no image key", () => {
      assert.deepEqual(restore({})?.image, seed);
    });

    it("keeps a saved look image over the packaged one", () => {
      assert.deepEqual(restore({ image: look })?.image, look);
    });
  });
});

describe("ic-lora recipe Restore", () => {
  const definition = createIcLoraRecipeDefinition(getIcLoraRecipe("restore"));
  const clip = { ...definition.defaults, video: { assetId: "clip" } };
  const small = { ...context(), videoWidth: 480, videoHeight: 270 };

  it("keeps a stored 1080p cell for a 270p source", () => {
    assert.equal(
      definition.normalize({ ...clip, resolution: "1080p" }, small).resolution,
      "1080p",
    );
    assert.equal(
      definition.toCreateBody({ ...clip, resolution: "1080p" }, small).params.resolution,
      "1080p",
    );
  });

  it("keeps the 720p default for a source below it", () => {
    assert.equal(definition.normalize(clip, small).resolution, "720p");
  });

  it("tags the cell closest to the source as Original", () => {
    const form = definition.form(clip, small);
    const field = form.fields.find((item) => item.dataKey === "resolution");
    assert.ok(field && field.kind === "options");
    if (field?.kind !== "options") return;
    assert.deepEqual(
      field.options.map((option) => option.label),
      ["270p (Original)", "360p", "540p", "720p", "1080p"],
    );
  });
});
