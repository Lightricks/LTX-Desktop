import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type {
  OfferingId,
  VideoGenerationAspectRatio,
  VideoGenerationDuration,
  VideoGenerationFps,
  VideoGenerationResolution,
} from "../../lib/video-generation-model-specs.ts";
import { specFixture, specFixtureBothOfferings } from "../screens/Feature/definitions/specFixture.ts";
import type { AssetRef, FeatureValues } from "../screens/Feature/types.ts";
import {
  VIDEO_SETTING_DURATIONS,
  VIDEO_SETTING_FPS_VALUES,
  VIDEO_SETTING_MODELS,
  VIDEO_SETTING_RESOLUTIONS,
  applyVideoSettingsChange,
  createVideoSettingsOptionFields,
  isSupportedVideoDuration,
  isSupportedVideoFps,
  isSupportedVideoModel,
  isSupportedVideoResolution,
} from "./videoSettingsControls.ts";

type LandscapeSchema = {
  prompt: { kind: "textarea" };
  model: { kind: "options"; value: OfferingId };
  aspectRatio: { kind: "options"; value: VideoGenerationAspectRatio };
  resolution: { kind: "options"; value: VideoGenerationResolution };
  duration: { kind: "options"; value: VideoGenerationDuration };
  fps: { kind: "options"; value: VideoGenerationFps };
};

type AutoSchema = {
  prompt: { kind: "textarea" };
  model: { kind: "options"; value: OfferingId };
  aspectRatio: { kind: "options"; value: "auto" | "16:9" | "9:16" };
  resolution: { kind: "options"; value: VideoGenerationResolution };
  duration: { kind: "options"; value: VideoGenerationDuration };
  fps: { kind: "options"; value: VideoGenerationFps };
  startFrame: { kind: "image-asset"; value: AssetRef | null };
  endFrame: { kind: "image-asset"; value: AssetRef | null };
};

const LOCAL_ASPECT_OPTIONS = [
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
];
const AUTO_ASPECT_OPTIONS = [
  { value: "auto", label: "Auto" },
  ...LOCAL_ASPECT_OPTIONS,
];
const LANDSCAPE_DEFAULTS: FeatureValues<LandscapeSchema> = {
  prompt: "example",
  model: "ltx-2.5-fast",
  aspectRatio: "16:9",
  resolution: "540p",
  duration: 8,
  fps: 24,
};
const AUTO_DEFAULTS: FeatureValues<AutoSchema> = {
  prompt: "drums",
  model: "ltx-2.5-fast",
  aspectRatio: "auto",
  resolution: "540p",
  duration: 8,
  fps: 24,
  startFrame: { assetId: "start-1" },
  endFrame: null,
};

function landscapeValues(
  overrides: Partial<FeatureValues<LandscapeSchema>> = {},
): FeatureValues<LandscapeSchema> {
  return { ...LANDSCAPE_DEFAULTS, ...overrides };
}

function autoValues(
  overrides: Partial<FeatureValues<AutoSchema>> = {},
): FeatureValues<AutoSchema> {
  return { ...AUTO_DEFAULTS, ...overrides };
}

function landscapeFields(
  overrides: Partial<FeatureValues<LandscapeSchema>> = {},
) {
  return createVideoSettingsOptionFields<LandscapeSchema>(
    landscapeValues(overrides),
    specFixture(),
  );
}

function optionValues(
  fields: readonly { dataKey: string; options: readonly { value: string | number }[] }[],
  dataKey: "model" | "aspectRatio" | "resolution" | "duration" | "fps",
) {
  return fields.find((field) => field.dataKey === dataKey)?.options.map((option) => option.value);
}

describe("video settings domains", () => {
  it("owns the shared catalogs and accepts only catalog members", () => {
    assert.deepEqual(VIDEO_SETTING_MODELS, ["ltx-2.5-fast", "ltx-2.3-fast"]);
    assert.deepEqual(VIDEO_SETTING_RESOLUTIONS, [
      "270p",
      "360p",
      "540p",
      "720p",
      "1080p",
      "1440p",
      "2160p",
    ]);
    assert.deepEqual(VIDEO_SETTING_FPS_VALUES, [24, 25, 48, 50]);
    assert.deepEqual(VIDEO_SETTING_DURATIONS, [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]);

    const membership = [
      [isSupportedVideoModel("ltx-2.5-fast"), true],
      [isSupportedVideoModel("pro-2.5"), false],
      [isSupportedVideoModel("unknown-model"), false],
      [isSupportedVideoResolution("1080p"), true],
      [isSupportedVideoResolution("4320p"), false],
      [isSupportedVideoFps(50), true],
      [isSupportedVideoFps(30), false],
      [isSupportedVideoDuration(8), true],
      [isSupportedVideoDuration(7), false],
    ] as const;
    for (const [actual, expected] of membership) {
      assert.equal(actual, expected);
    }
  });
});

describe("createVideoSettingsOptionFields", () => {
  it("reads aspect options from the spec cell without inventing Auto", () => {
    const fields = landscapeFields();
    assert.deepEqual(
      fields.map((field) => field.dataKey),
      ["model", "aspectRatio", "resolution", "duration", "fps"],
    );
    assert.deepEqual(
      fields.find((field) => field.dataKey === "aspectRatio")?.options,
      LOCAL_ASPECT_OPTIONS,
    );
    assert.equal(
      fields.some((field) => field.options.some((option) => option.value === "auto")),
      false,
    );
  });

  it("prepends Auto for image-to-video and keeps the spec cell behind it", () => {
    const fields = createVideoSettingsOptionFields<AutoSchema>(
      autoValues({ aspectRatio: "auto" }),
      specFixture(),
      { includeAuto: true },
    );
    assert.deepEqual(
      fields.find((field) => field.dataKey === "aspectRatio")?.options,
      AUTO_ASPECT_OPTIONS,
    );
    assert.deepEqual(optionValues(fields, "duration"), [2, 3, 4, 5, 6, 8, 10, 20]);
    assert.deepEqual(optionValues(fields, "fps"), [24, 25, 48, 50]);
  });

  it("lists resolution, duration, and fps in ascending order whatever the spec order", () => {
    const specs = specFixture();
    const offering = specs.downloaded_local_models[0];
    const cells = offering.spec.supported_resolutions_durations;
    const shuffled = {
      "1080p": cells["1080p"],
      "540p": cells["540p"],
      "720p": cells["720p"],
    };
    const fields = createVideoSettingsOptionFields<LandscapeSchema>(landscapeValues(), {
      ...specs,
      downloaded_local_models: [
        {
          ...offering,
          spec: { ...offering.spec, supported_resolutions_durations: shuffled },
        },
      ],
    });
    assert.deepEqual(optionValues(fields, "resolution"), ["540p", "720p", "1080p"]);
    assert.deepEqual(optionValues(fields, "duration"), [2, 3, 4, 5, 6, 8, 10, 20]);
    assert.deepEqual(optionValues(fields, "fps"), [24, 25, 48, 50]);
  });

  it("narrows duration and fps from the local model catalog", () => {
    const fastFields = landscapeFields();
    assert.deepEqual(optionValues(fastFields, "model"), ["ltx-2.5-fast"]);
    assert.deepEqual(
      fastFields
        .find((field) => field.dataKey === "model")
        ?.options.map((option) => option.label),
      ["LTX 2.5 Fast"],
    );
    assert.deepEqual(optionValues(fastFields, "duration"), [2, 3, 4, 5, 6, 8, 10, 20]);
    assert.deepEqual(optionValues(fastFields, "fps"), [24, 25, 48, 50]);
  });

  it("lists downloaded offerings newest first, not the active pipeline row", () => {
    const fields = createVideoSettingsOptionFields<LandscapeSchema>(
      landscapeValues(),
      specFixtureBothOfferings(),
      {},
    );
    assert.deepEqual(optionValues(fields, "model"), [
      "ltx-2.5-fast",
      "ltx-2.3-fast",
    ]);
    assert.deepEqual(
      fields
        .find((field) => field.dataKey === "model")
        ?.options.map((option) => option.label),
      ["LTX 2.5 Fast", "LTX 2.3 Fast"],
    );
  });

  it("is empty when nothing is downloaded even if local_models advertises fast", () => {
    const fields = createVideoSettingsOptionFields<LandscapeSchema>(
      landscapeValues(),
      { api_models: [], local_models: specFixture().local_models, downloaded_local_models: [] },
      {},
    );
    assert.deepEqual(optionValues(fields, "model"), []);
  });
});

describe("applyVideoSettingsChange", () => {
  it("retargets duration and fps on distinct branches without dropping extra fields", () => {
    const afterDuration = applyVideoSettingsChange(
      landscapeValues({
        model: "ltx-2.5-fast",
        resolution: "1080p",
        fps: 24,
        duration: 6,
        prompt: "keep me",
      }),
      { kind: "options", fieldId: "duration", dataKey: "duration", value: 8 },
      specFixture(),
      LANDSCAPE_DEFAULTS,
    );
    assert.equal(afterDuration.duration, 8);
    assert.equal(afterDuration.resolution, "540p");
    assert.equal(afterDuration.fps, 24);
    assert.equal(afterDuration.prompt, "keep me");
    assert.equal(afterDuration.aspectRatio, "16:9");

    const afterFps = applyVideoSettingsChange(
      landscapeValues({
        model: "ltx-2.5-fast",
        resolution: "1080p",
        fps: 24,
        duration: 10,
      }),
      { kind: "options", fieldId: "fps", dataKey: "fps", value: 24 },
      specFixture(),
      LANDSCAPE_DEFAULTS,
    );
    assert.equal(afterFps.fps, 24);
    assert.equal(afterFps.resolution, "1080p");
    assert.equal(afterFps.duration, 10);
  });

  it("persists Auto through aspect edits and later clamps", () => {
    const afterAspect = applyVideoSettingsChange(
      autoValues({ aspectRatio: "16:9" }),
      { kind: "options", fieldId: "aspectRatio", dataKey: "aspectRatio", value: "auto" },
      specFixture(),
      AUTO_DEFAULTS,
    );
    const afterDuration = applyVideoSettingsChange(
      afterAspect,
      { kind: "options", fieldId: "duration", dataKey: "duration", value: 8 },
      specFixture(),
      AUTO_DEFAULTS,
    );
    assert.equal(afterAspect.aspectRatio, "auto");
    assert.equal(afterDuration.aspectRatio, "auto");
    assert.deepEqual(afterDuration.startFrame, { assetId: "start-1" });
    assert.equal(afterDuration.endFrame, null);
  });

  it("applies prompt edits without retargeting video fields", () => {
    const current = landscapeValues({
      model: "ltx-2.5-fast",
      resolution: "720p",
      fps: 24,
      duration: 6,
    });
    const next = applyVideoSettingsChange(
      current,
      { kind: "textarea", fieldId: "prompt", dataKey: "prompt", value: "new prompt" },
      specFixture(),
      LANDSCAPE_DEFAULTS,
    );
    assert.equal(next.prompt, "new prompt");
    assert.equal(next.model, current.model);
    assert.equal(next.resolution, current.resolution);
    assert.equal(next.fps, current.fps);
    assert.equal(next.duration, current.duration);
  });
});
