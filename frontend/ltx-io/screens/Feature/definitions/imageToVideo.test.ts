import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  resolveFeatureFormValues,
  resolveFeatureSeedValues,
  type FeatureFieldChange,
} from "../types.ts";
import {
  applyOptionFieldOrder,
  groupConsecutiveOptionFields,
} from "../featureFormDisplay.ts";
import { specFixture, specFixtureBothOfferings } from "./specFixture.ts";
import {
  IMAGE_TO_VIDEO_DEFAULTS,
  IMAGE_TO_VIDEO_EXAMPLE_PROMPT,
  fromGeneration,
  imageToVideoDefinition,
  toCreateBody,
  validateImageToVideo,
  type ImageToVideoContext,
  type ImageToVideoSchema,
  type ImageToVideoValues,
} from "./imageToVideo.ts";
import { orderImageToVideoOptionFields } from "../imageToVideoOptionFieldOrder.ts";

const seedAsset = { assetId: "seed-asset" };
const startAsset = { assetId: "start-1" };
const endAsset = { assetId: "end-1" };

function context(
  overrides: Partial<ImageToVideoContext> = {},
): ImageToVideoContext {
  return { specs: specFixture(), seed: seedAsset, ...overrides };
}

function values(overrides: Partial<ImageToVideoValues> = {}): ImageToVideoValues {
  return { ...IMAGE_TO_VIDEO_DEFAULTS, startFrame: startAsset, ...overrides };
}

describe("imageToVideo definition", () => {
  it("seeds the example prompt, Auto aspect, downloaded Fast, 24fps, 540p, and 8s", () => {
    assert.equal(imageToVideoDefinition.id, "image-to-video");
    assert.equal(imageToVideoDefinition.title, "Image to Video");
    assert.deepEqual(imageToVideoDefinition.defaults, {
      prompt: IMAGE_TO_VIDEO_EXAMPLE_PROMPT,
      model: "ltx-2.5-fast",
      aspectRatio: "auto",
      resolution: "540p",
      duration: 8,
      fps: 24,
      startFrame: null,
      endFrame: null,
    });
    assert.equal(IMAGE_TO_VIDEO_DEFAULTS.model, "ltx-2.5-fast");
    assert.equal(IMAGE_TO_VIDEO_DEFAULTS.resolution, "540p");
    assert.equal(IMAGE_TO_VIDEO_DEFAULTS.fps, 24);
    assert.equal(IMAGE_TO_VIDEO_DEFAULTS.duration, 8);
    assert.equal(IMAGE_TO_VIDEO_EXAMPLE_PROMPT, "A young man plays the drums.");
  });

  it("renders Start Frame, End Frame, prompt, then video settings", () => {
    const fields = imageToVideoDefinition.form(values(), context()).fields;
    assert.deepEqual(
      fields.map((field) => ({ kind: field.kind, id: field.id, dataKey: field.dataKey })),
      [
        { kind: "image-asset", id: "startFrame", dataKey: "startFrame" },
        { kind: "image-asset", id: "endFrame", dataKey: "endFrame" },
        { kind: "textarea", id: "prompt", dataKey: "prompt" },
        { kind: "options", id: "model", dataKey: "model" },
        { kind: "options", id: "aspectRatio", dataKey: "aspectRatio" },
        { kind: "options", id: "resolution", dataKey: "resolution" },
        { kind: "options", id: "duration", dataKey: "duration" },
        { kind: "options", id: "fps", dataKey: "fps" },
      ],
    );
    assert.equal(fields[0]?.kind === "image-asset" ? fields[0].label : undefined, "Start Frame");
    assert.equal(fields[1]?.kind === "image-asset" ? fields[1].label : undefined, "End Frame");
  });

  it("keeps image slots and the prompt, then clusters settings", () => {
    const form = imageToVideoDefinition.form(values(), context());
    const displayFields = applyOptionFieldOrder(
      form.fields,
      orderImageToVideoOptionFields,
    );
    const groups = groupConsecutiveOptionFields(displayFields);

    assert.deepEqual(
      displayFields.map((field) => ({
        kind: field.kind,
        dataKey: field.dataKey,
      })),
      [
        { kind: "image-asset", dataKey: "startFrame" },
        { kind: "image-asset", dataKey: "endFrame" },
        { kind: "textarea", dataKey: "prompt" },
        { kind: "options", dataKey: "model" },
        { kind: "options", dataKey: "aspectRatio" },
        { kind: "options", dataKey: "resolution" },
        { kind: "options", dataKey: "duration" },
        { kind: "options", dataKey: "fps" },
      ],
    );
    assert.deepEqual(
      groups.map((group) =>
        group.kind === "options"
          ? {
              kind: group.kind,
              dataKeys: group.fields.map((field) => field.dataKey),
            }
          : { kind: group.kind, dataKey: group.field.dataKey },
      ),
      [
        { kind: "image-asset", dataKey: "startFrame" },
        { kind: "image-asset", dataKey: "endFrame" },
        { kind: "textarea", dataKey: "prompt" },
        {
          kind: "options",
          dataKeys: ["model", "aspectRatio", "resolution", "duration", "fps"],
        },
      ],
    );
  });

  it("offers Auto plus the local spec cell", () => {
    const aspect = imageToVideoDefinition
      .form(values(), context())
      .fields.find((field) => field.kind === "options" && field.dataKey === "aspectRatio");
    assert.equal(aspect?.kind, "options");
    assert.deepEqual(
      aspect?.kind === "options" ? aspect.options : undefined,
      [
        { value: "auto", label: "Auto" },
        { value: "16:9", label: "16:9" },
        { value: "9:16", label: "9:16" },
        {
          value: "21:9",
          label: "21:9",
          warning:
            "The model was trained on 16:9 and 9:16, other ratios may produce lower quality results.",
        },
        {
          value: "3:2",
          label: "3:2",
          warning:
            "The model was trained on 16:9 and 9:16, other ratios may produce lower quality results.",
        },
        {
          value: "4:3",
          label: "4:3",
          warning:
            "The model was trained on 16:9 and 9:16, other ratios may produce lower quality results.",
        },
        {
          value: "1:1",
          label: "1:1",
          warning:
            "The model was trained on 16:9 and 9:16, other ratios may produce lower quality results.",
        },
        {
          value: "4:5",
          label: "4:5",
          warning:
            "The model was trained on 16:9 and 9:16, other ratios may produce lower quality results.",
        },
      ],
    );
  });

  it("requires Start Frame and allows a blank prompt with Start Frame", () => {
    assert.deepEqual(validateImageToVideo(values({ startFrame: null, prompt: "ok" })), [
      {
        id: "start-frame-required",
        fieldId: "startFrame",
        message: "Start Frame is required.",
      },
    ]);
    assert.deepEqual(validateImageToVideo(values({ prompt: "  " })), []);
    assert.deepEqual(validateImageToVideo(values({ prompt: "" })), []);
  });

  it("rejects End Frame when Start Frame is missing", () => {
    assert.deepEqual(
      validateImageToVideo(
        values({ startFrame: null, endFrame: endAsset, prompt: "ok" }),
      ),
      [
        {
          id: "start-frame-required",
          fieldId: "startFrame",
          message: "Start Frame is required.",
        },
        {
          id: "end-frame-requires-start",
          fieldId: "endFrame",
          message: "End Frame requires a Start Frame.",
        },
      ],
    );
    assert.deepEqual(
      validateImageToVideo(values({ endFrame: endAsset })),
      [],
    );
  });

  it("maps only asset IDs into the create payload and preserves Auto", () => {
    const body = toCreateBody(
      values({
        prompt: "  drums  ",
        aspectRatio: "auto",
        endFrame: endAsset,
      }),
    );
    assert.deepEqual(body, {
      contract_version: 1,
      inputs: {
        startFrame: { assetId: "start-1" },
        endFrame: { assetId: "end-1" },
      },
      params: {
        prompt: "drums",
        model: "ltx-2.5-fast",
        aspectRatio: "auto",
        resolution: "540p",
        duration: 8,
        fps: 24,
        cameraMotion: "none",
        negativePrompt: "",
        promptProvenance: "typed",
      },
    });
    assert.equal("path" in body.inputs.startFrame, false);
    assert.equal(JSON.stringify(body).includes("/"), false);
    assert.equal(JSON.stringify(body).includes("blob:"), false);
  });

  it("sends explicit 16:9 and 9:16 overrides without converting Auto", () => {
    assert.equal(toCreateBody(values({ aspectRatio: "16:9" })).params.aspectRatio, "16:9");
    assert.equal(toCreateBody(values({ aspectRatio: "9:16" })).params.aspectRatio, "9:16");
    assert.equal(toCreateBody(values({ aspectRatio: "auto" })).params.aspectRatio, "auto");
  });

  it("omits End Frame as null when the optional slot is empty", () => {
    assert.equal(toCreateBody(values({ endFrame: null })).inputs.endFrame, null);
  });
});

describe("imageToVideo fromGeneration", () => {
  it("restores both input slots from spec.inputs asset IDs", () => {
    const parsed = fromGeneration({
      params: {
        prompt: "a drummer",
        model: "ltx-2.5-fast",
        aspectRatio: "auto",
        resolution: "720p",
        duration: 8,
        fps: 24,
      },
      inputs: {
        startFrame: { assetId: "from-start" },
        endFrame: { assetId: "from-end" },
      },
    });
    assert.deepEqual(parsed, {
      prompt: "a drummer",
      model: "ltx-2.5-fast",
      aspectRatio: "auto",
      resolution: "720p",
      duration: 8,
      fps: 24,
      startFrame: { assetId: "from-start" },
      endFrame: { assetId: "from-end" },
    });
    assert.equal(parsed && "path" in parsed.startFrame, false);
  });

  it("restores a missing End Frame as null and keeps Auto", () => {
    const parsed = fromGeneration({
      params: {
        prompt: "",
        aspectRatio: "auto",
      },
      inputs: {
        startFrame: { assetId: "only-start" },
      },
    });
    assert.equal(parsed?.startFrame?.assetId, "only-start");
    assert.equal(parsed?.endFrame, null);
    assert.equal(parsed?.aspectRatio, "auto");
    assert.equal(parsed?.prompt, "");
  });

  it("returns null when spec.inputs.startFrame is missing", () => {
    assert.equal(fromGeneration({ params: { prompt: "x" } }), null);
    assert.equal(fromGeneration({ inputs: {} }), null);
    assert.equal(
      fromGeneration({
        params: { prompt: "x" },
        inputs: { endFrame: { assetId: "end" } },
      }),
      null,
    );
  });
});

describe("imageToVideo seed and hydration paths", () => {
  it("seeds Start Frame from initialValues when the form has never been persisted", () => {
    const seeded = resolveFeatureSeedValues(imageToVideoDefinition, {
      context: context(),
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: undefined,
    });
    assert.deepEqual(seeded.startFrame, seedAsset);
    assert.equal(seeded.prompt, IMAGE_TO_VIDEO_EXAMPLE_PROMPT);
    assert.equal(seeded.aspectRatio, "auto");
  });

  it("does not reinsert the seed after an explicitly persisted clear", () => {
    const cleared = values({ startFrame: null, prompt: "kept" });
    const next = resolveFeatureSeedValues(imageToVideoDefinition, {
      context: context(),
      hasStoredValues: true,
      storedValues: cleared,
      lastGenerationSpec: undefined,
    });
    assert.equal(next.startFrame, null);
    assert.equal(next.prompt, "kept");
  });

  it("keeps a reselected Start Frame instead of the packaged seed", () => {
    const stored = values({ startFrame: { assetId: "user-asset" } });
    const next = resolveFeatureSeedValues(imageToVideoDefinition, {
      context: context(),
      hasStoredValues: true,
      storedValues: stored,
      lastGenerationSpec: undefined,
    });
    assert.deepEqual(next.startFrame, { assetId: "user-asset" });
  });

  it("hydrates prior-generation inputs before applying the seed", () => {
    const next = resolveFeatureSeedValues(imageToVideoDefinition, {
      context: context(),
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: {
        params: { prompt: "from history", aspectRatio: "9:16" },
        inputs: {
          startFrame: { assetId: "history-start" },
          endFrame: { assetId: "history-end" },
        },
      },
    });
    assert.deepEqual(next.startFrame, { assetId: "history-start" });
    assert.deepEqual(next.endFrame, { assetId: "history-end" });
    assert.equal(next.prompt, "from history");
    assert.equal(next.aspectRatio, "9:16");
  });
});

describe("imageToVideo setting fields", () => {
  it("uses Studio labels and forcePicker on model and duration", () => {
    const fields = orderImageToVideoOptionFields(
      imageToVideoDefinition
        .form(values(), context())
        .fields.filter((field) => field.kind === "options"),
    );

    assert.deepEqual(
      fields.map((field) => ({
        dataKey: field.dataKey,
        label: field.label,
        forcePicker: field.forcePicker ?? false,
        maxOptionCount: field.maxOptionCount,
      })),
      [
        {
          dataKey: "model",
          label: "Model",
          forcePicker: true,
          maxOptionCount: undefined,
        },
        {
          dataKey: "aspectRatio",
          label: "Aspect Ratio",
          forcePicker: false,
          maxOptionCount: undefined,
        },
        {
          dataKey: "resolution",
          label: "Resolution",
          forcePicker: false,
          maxOptionCount: 7,
        },
        {
          dataKey: "duration",
          label: "Duration",
          forcePicker: true,
          maxOptionCount: undefined,
        },
        {
          dataKey: "fps",
          label: "Frame Rate",
          forcePicker: false,
          maxOptionCount: 4,
        },
      ],
    );
  });

  it("routes duration changes through the video context retarget policy", () => {
    const durationChange: FeatureFieldChange<ImageToVideoSchema> = {
      kind: "options",
      fieldId: "duration",
      dataKey: "duration",
      value: 8,
    };
    const next = imageToVideoDefinition.applyChange(
      values({
        model: "ltx-2.5-fast",
        resolution: "1080p",
        fps: 24,
        duration: 6,
        aspectRatio: "auto",
      }),
      durationChange,
      context(),
    );

    assert.equal(next.duration, 8);
    assert.equal(next.resolution, "540p");
    assert.equal(next.fps, 24);
    assert.equal(next.aspectRatio, "auto");
    assert.deepEqual(next.startFrame, startAsset);
  });

  it("persists an Auto aspect change without replacing it with a closest ratio", () => {
    const next = imageToVideoDefinition.applyChange(
      values({ aspectRatio: "16:9" }),
      { kind: "options", fieldId: "aspectRatio", dataKey: "aspectRatio", value: "auto" },
      context(),
    );
    assert.equal(next.aspectRatio, "auto");
    assert.equal(
      imageToVideoDefinition.normalize(next, context()).aspectRatio,
      "auto",
    );
  });

  it("stores only assetId when applying Start or End Frame changes", () => {
    const startChange: FeatureFieldChange<ImageToVideoSchema> = {
      kind: "image-asset",
      fieldId: "startFrame",
      dataKey: "startFrame",
      value: { assetId: "uploaded-start" },
    };

    const withStart = imageToVideoDefinition.applyChange(
      values({ startFrame: null }),
      startChange,
      context(),
    );
    assert.deepEqual(withStart.startFrame, { assetId: "uploaded-start" });

    const cleared = imageToVideoDefinition.applyChange(
      withStart,
      { kind: "image-asset", fieldId: "startFrame", dataKey: "startFrame", value: null },
      context(),
    );
    assert.equal(cleared.startFrame, null);
  });

  it("restores last-generation frames but maps unsupported model to the Settings offering", () => {
    const seeded = resolveFeatureSeedValues(imageToVideoDefinition, {
      context: context({ specs: specFixtureBothOfferings("LTX 2.3 Fast") }),
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: {
        params: {
          prompt: "a drummer",
          model: "fast",
          aspectRatio: "auto",
          resolution: "720p",
          duration: 8,
          fps: 24,
        },
        inputs: { startFrame: { assetId: "from-start" } },
      },
    });
    assert.equal(seeded.model, "ltx-2.3-fast");
    assert.deepEqual(seeded.startFrame, { assetId: "from-start" });
  });

  it("shows the Settings-matching offering on first paint before the store write", () => {
    const displayed = resolveFeatureFormValues(imageToVideoDefinition, {
      context: context({ specs: specFixtureBothOfferings("LTX 2.3 Fast") }),
      contextReady: true,
      storedValues: undefined,
      generationsFetched: true,
      lastGenerationSpec: undefined,
    });
    assert.equal(displayed.model, "ltx-2.3-fast");
  });

  it("resetValues picks the Settings-matching offering and clears frames", () => {
    const reset = imageToVideoDefinition.resetValues?.({
      specs: specFixtureBothOfferings("LTX 2.3 Fast"),
      seed: seedAsset,
    });
    assert.equal(reset?.model, "ltx-2.3-fast");
    assert.equal(reset?.startFrame, null);
    assert.equal(reset?.endFrame, null);
    assert.equal(reset?.prompt, IMAGE_TO_VIDEO_EXAMPLE_PROMPT);
  });
});
