import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  clampVideoFields,
  isLocalVideoSelectionReady,
} from "../../../lib/videoFieldPolicy.ts";
import {
  applyOptionFieldOrder,
  groupConsecutiveOptionFields,
} from "../featureFormDisplay.ts";
import { orderTextToVideoOptionFields } from "../textToVideoOptionFieldOrder.ts";
import {
  resolveFeatureFormValues,
  resolveFeatureSeedValues,
  type FeatureFieldChange,
} from "../types.ts";
import { specFixture, specFixtureBothOfferings } from "./specFixture.ts";
import { defaultOfferingId } from "../../../../lib/video-generation-model-specs.ts";
import {
  TEXT_TO_VIDEO_DEFAULTS,
  TEXT_TO_VIDEO_EXAMPLE_PROMPT,
  fromGeneration,
  textToVideoDefinition,
  toCreateBody,
  validateTextToVideo,
  type TextToVideoSchema,
  type TextToVideoValues,
} from "./textToVideo.ts";

function values(overrides: Partial<TextToVideoValues> = {}): TextToVideoValues {
  return { ...TEXT_TO_VIDEO_DEFAULTS, ...overrides };
}

describe("textToVideo definition", () => {
  it("seeds the example prompt + local-valid fast / 540p / 24fps / 8s", () => {
    assert.deepEqual(textToVideoDefinition.defaults, {
      prompt: TEXT_TO_VIDEO_EXAMPLE_PROMPT,
      model: "ltx-2.5-fast",
      aspectRatio: "16:9",
      resolution: "540p",
      duration: 8,
      fps: 24,
    });
    const promptField = textToVideoDefinition.form(values(), {
      specs: specFixture(),
    }).fields[0];
    assert.equal(promptField?.kind, "textarea");
    assert.equal(
      promptField?.kind === "textarea" ? promptField.placeholder : undefined,
      "The woman sips from a cup of coffee...",
    );
    assert.match(TEXT_TO_VIDEO_EXAMPLE_PROMPT, /dalmatian/i);
  });

  it("hydrates store values from a generation spec", () => {
    const parsed = fromGeneration({
      params: {
        prompt: "a woman sips coffee",
        model: "ltx-2.5-fast",
        aspectRatio: "9:16",
        resolution: "720p",
        duration: 8,
        fps: 24,
      },
    });
    assert.deepEqual(parsed, {
      prompt: "a woman sips coffee",
      model: "ltx-2.5-fast",
      aspectRatio: "9:16",
      resolution: "720p",
      duration: 8,
      fps: 24,
    });
  });

  it("returns null when spec.params is missing", () => {
    assert.equal(fromGeneration({}), null);
    assert.equal(fromGeneration({ params: { model: "fast" } }), null);
  });

  it("requires a prompt", () => {
    assert.deepEqual(validateTextToVideo(values({ prompt: "  " })), [
      {
        id: "prompt-required",
        fieldId: "prompt",
        message: "Prompt is required.",
      },
    ]);
    assert.deepEqual(validateTextToVideo(values({ prompt: "ok" })), []);
  });

  it("builds a create body with hidden cameraMotion, negativePrompt, and loras", () => {
    assert.deepEqual(toCreateBody(values({ prompt: "  hello  " })), {
      contract_version: 1,
      params: {
        prompt: "hello",
        model: "ltx-2.5-fast",
        aspectRatio: "16:9",
        resolution: "540p",
        duration: 8,
        fps: 24,
        cameraMotion: "none",
        negativePrompt: "",
        loras: [],
        promptProvenance: "typed",
      },
    });
  });
});

describe("textToVideo setting fields", () => {
  it("renders prompt then model, aspect, resolution, duration, and fps", () => {
    const fields = textToVideoDefinition.form(values(), { specs: specFixture() }).fields;
    assert.deepEqual(
      fields.map((field) => ({ kind: field.kind, id: field.id, dataKey: field.dataKey })),
      [
        { kind: "textarea", id: "prompt", dataKey: "prompt" },
        { kind: "options", id: "model", dataKey: "model" },
        { kind: "options", id: "aspectRatio", dataKey: "aspectRatio" },
        { kind: "options", id: "resolution", dataKey: "resolution" },
        { kind: "options", id: "duration", dataKey: "duration" },
        { kind: "options", id: "fps", dataKey: "fps" },
      ],
    );
  });

  it("offers the local spec cell with the trained pair first", () => {
    const aspect = textToVideoDefinition
      .form(values(), { specs: specFixture() })
      .fields.find((field) => field.kind === "options" && field.dataKey === "aspectRatio");
    assert.equal(aspect?.kind, "options");
    assert.deepEqual(
      aspect?.kind === "options" ? aspect.options : undefined,
      [
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

  it("uses Studio labels and forcePicker on model and duration", () => {
    const fields = orderTextToVideoOptionFields(
      textToVideoDefinition
        .form(values(), { specs: specFixture() })
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
    assert.deepEqual(
      fields
        .find((field) => field.dataKey === "duration")
        ?.options.map((option) => option.value),
      [2, 3, 4, 5, 6, 8, 10, 20],
    );
    assert.deepEqual(
      fields
        .find((field) => field.dataKey === "fps")
        ?.options.map((option) => option.value),
      [24, 25, 48, 50],
    );
  });

  it("reorders option slots, keeps the prompt textarea, and clusters settings", () => {
    const form = textToVideoDefinition.form(values(), { specs: specFixture() });
    const displayFields = applyOptionFieldOrder(
      form.fields,
      orderTextToVideoOptionFields,
    );
    const groups = groupConsecutiveOptionFields(displayFields);

    assert.deepEqual(
      displayFields.map((field) => ({
        kind: field.kind,
        dataKey: field.dataKey,
      })),
      [
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
        { kind: "textarea", dataKey: "prompt" },
        {
          kind: "options",
          dataKeys: ["model", "aspectRatio", "resolution", "duration", "fps"],
        },
      ],
    );
  });

  it("routes duration changes through the video context retarget policy", () => {
    const durationChange: FeatureFieldChange<TextToVideoSchema> = {
      kind: "options",
      fieldId: "duration",
      dataKey: "duration",
      value: 8,
    };

    const next = textToVideoDefinition.applyChange(
      values({
        model: "ltx-2.5-fast",
        resolution: "1080p",
        fps: 24,
        duration: 6,
      }),
      durationChange,
      { specs: specFixture() },
    );

    assert.equal(next.duration, 8);
    assert.equal(next.resolution, "540p");
    assert.equal(next.fps, 24);
    assert.equal(next.prompt, TEXT_TO_VIDEO_EXAMPLE_PROMPT);
  });
});

describe("local video generate readiness", () => {
  const specs = specFixture();

  it("is not ready before specs resolve, so Generate can't submit anything", () => {
    assert.equal(isLocalVideoSelectionReady(TEXT_TO_VIDEO_DEFAULTS, null), false);
    assert.equal(
      isLocalVideoSelectionReady(TEXT_TO_VIDEO_DEFAULTS, undefined),
      false,
    );
  });

  it("is not ready while a value is still unclamped after specs resolve", () => {
    // 1080p locally only runs [10]; duration 8 there isn't submittable until the
    // clamp re-selects, so the raw selection must be reported as not ready.
    assert.equal(
      isLocalVideoSelectionReady(
        values({ resolution: "1080p", duration: 8 }),
        specs,
      ),
      false,
    );
  });

  it("is ready once values are clamped to the local catalog", () => {
    const clamped = clampVideoFields(
      TEXT_TO_VIDEO_DEFAULTS,
      specs,
      TEXT_TO_VIDEO_DEFAULTS,
    );
    assert.equal(isLocalVideoSelectionReady(clamped, specs), true);
  });
});

describe("local video unavailable signal", () => {
  it("is available for a downloaded catalog", () => {
    assert.equal(
      textToVideoDefinition.isUnavailable?.(values(), { specs: specFixture() }),
      false,
    );
  });

  it("is unavailable when nothing is downloaded even if local_models advertises fast", () => {
    assert.equal(
      textToVideoDefinition.isUnavailable?.(values(), {
        specs: {
          api_models: [],
          local_models: specFixture().local_models,
          downloaded_local_models: [],
        },
      }),
      true,
    );
  });

  it("is unavailable when the downloaded cell has no durations", () => {
    assert.equal(
      textToVideoDefinition.isUnavailable?.(values(), {
        specs: {
          api_models: [],
          local_models: specFixture().local_models,
          downloaded_local_models: [
            {
              model: "ltx-2.5-fast",
              pipeline: "fast",
              spec: {
                display_name: "LTX 2.5 Fast",
                supported_resolutions_durations: {
                  "720p": { fps_to_durations: { "24": [] }, aspect_ratios: ["16:9"] },
                },
              },
            },
          ],
        },
      }),
      true,
    );
  });
});

describe("textToVideo offering default", () => {
  it("seeds the offering that matches Settings' active local row", () => {
    const seeded = resolveFeatureSeedValues(textToVideoDefinition, {
      context: { specs: specFixtureBothOfferings("LTX 2.3 Fast") },
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: undefined,
    });
    assert.equal(seeded.model, "ltx-2.3-fast");
  });

  it("returns null when nothing is downloaded", () => {
    assert.equal(
      defaultOfferingId({
        api_models: [],
        local_models: specFixture().local_models,
        downloaded_local_models: [],
      }),
      null,
    );
  });

  it("restores last-generation prompt but maps unsupported model to the Settings offering", () => {
    const seeded = resolveFeatureSeedValues(textToVideoDefinition, {
      context: { specs: specFixtureBothOfferings("LTX 2.3 Fast") },
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: {
        params: {
          prompt: "a woman sips coffee",
          model: "fast",
          aspectRatio: "9:16",
          resolution: "720p",
          duration: 8,
          fps: 24,
        },
      },
    });
    assert.equal(seeded.model, "ltx-2.3-fast");
    assert.equal(seeded.prompt, "a woman sips coffee");
    assert.equal(seeded.aspectRatio, "9:16");
  });

  it("keeps a last-generation offering id even when Settings is a different row", () => {
    const seeded = resolveFeatureSeedValues(textToVideoDefinition, {
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
    const displayed = resolveFeatureFormValues(textToVideoDefinition, {
      context: { specs: specFixtureBothOfferings("LTX 2.3 Fast") },
      contextReady: true,
      storedValues: undefined,
      generationsFetched: true,
      lastGenerationSpec: undefined,
    });
    assert.equal(displayed.model, "ltx-2.3-fast");
  });

  it("resetValues picks the Settings-matching offering, not hardcoded 2.5", () => {
    const reset = textToVideoDefinition.resetValues?.({
      specs: specFixtureBothOfferings("LTX 2.3 Fast"),
    });
    assert.equal(reset?.model, "ltx-2.3-fast");
    assert.equal(reset?.prompt, TEXT_TO_VIDEO_EXAMPLE_PROMPT);
  });

  it("applyChange falls back to the Settings-matching offering, not hardcoded 2.5", () => {
    const change: FeatureFieldChange<TextToVideoSchema> = {
      kind: "options",
      fieldId: "aspectRatio",
      dataKey: "aspectRatio",
      value: "9:16",
    };

    const next = textToVideoDefinition.applyChange(
      values({
        model: "fast" as TextToVideoValues["model"],
      }),
      change,
      { specs: specFixtureBothOfferings("LTX 2.3 Fast") },
    );

    assert.equal(next.model, "ltx-2.3-fast");
  });
});
