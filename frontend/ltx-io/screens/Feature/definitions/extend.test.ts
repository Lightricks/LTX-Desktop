import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { VideoGenerationModelSpecsResponse } from "../../../../lib/video-generation-model-specs.ts";
import { resolveFeatureSeedValues } from "../types.ts";
import { specFixture, specFixtureBothOfferings } from "./specFixture.ts";
import {
  EXTEND_DEFAULTS,
  EXTEND_EXAMPLE_PROMPT,
  extendDefinition,
  fromGeneration,
  hasAdvertisedExtend,
  toCreateBody,
  validateExtend,
  type ExtendContext,
  type ExtendValues,
} from "./extend.ts";

const videoAsset = { assetId: "video-1" };

const EXTEND_CAPS = {
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

function withExtendCaps(
  specs: VideoGenerationModelSpecsResponse,
): VideoGenerationModelSpecsResponse {
  return {
    ...specs,
    downloaded_local_models: specs.downloaded_local_models.map((item) => ({
      ...item,
      spec: { ...item.spec, capabilities: { ...EXTEND_CAPS } },
    })),
  };
}

const seedAsset = { assetId: "seed-video" };

function context(overrides: Partial<ExtendContext> = {}): ExtendContext {
  return {
    specs: withExtendCaps(specFixture()),
    seed: seedAsset,
    videoDurationSeconds: 8,
    videoDurationPending: false,
    videoWidth: null,
    videoHeight: null,
    videoFps: null,
    ...overrides,
  };
}

function values(overrides: Partial<ExtendValues> = {}): ExtendValues {
  return {
    ...EXTEND_DEFAULTS,
    video: videoAsset,
    ...overrides,
  };
}

describe("extend definition", () => {
  it("does not cap the source video length", () => {
    const form = extendDefinition.form(values(), context());
    const video = form.fields.find((field) => field.id === "video");
    assert.equal(video?.kind, "video-asset");
    if (video?.kind !== "video-asset") return;
    assert.equal(video.maxDurationSeconds, undefined);
    assert.deepEqual(
      validateExtend(values(), context({ videoDurationSeconds: 75 })),
      [],
    );
  });

  it("seeds offering from the Settings-matching download", () => {
    const seeded = resolveFeatureSeedValues(extendDefinition, {
      context: context({ specs: withExtendCaps(specFixtureBothOfferings()) }),
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: undefined,
    });
    assert.equal(seeded.model, "ltx-2.3-fast");
    assert.equal(seeded.prompt, EXTEND_EXAMPLE_PROMPT);
    assert.equal(seeded.duration, 4);
    assert.equal(seeded.mode, "end");
    assert.deepEqual(seeded.video, seedAsset);
  });

  it("lists prepend before extend and keeps extend selected by default", () => {
    const form = extendDefinition.form(values(), context());
    const mode = form.fields.find((field) => field.id === "mode");
    assert.equal(mode?.kind, "options");
    if (mode?.kind !== "options") return;
    assert.deepEqual(
      mode.options.map((option) => option.value),
      ["start", "end"],
    );
    assert.equal(EXTEND_DEFAULTS.mode, "end");
  });

  it("seeds Video from initialValues when the form has never been persisted", () => {
    const seeded = resolveFeatureSeedValues(extendDefinition, {
      context: context(),
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: undefined,
    });
    assert.deepEqual(seeded.video, seedAsset);
  });

  it("does not reinsert the seed after an explicitly persisted clear", () => {
    const cleared = values({ video: null, prompt: "kept" });
    const next = resolveFeatureSeedValues(extendDefinition, {
      context: context(),
      hasStoredValues: true,
      storedValues: cleared,
      lastGenerationSpec: undefined,
    });
    assert.equal(next.video, null);
    assert.equal(next.prompt, "kept");
  });

  it("keeps a reselected video instead of the packaged seed", () => {
    const stored = values({ video: { assetId: "user-asset" } });
    const next = resolveFeatureSeedValues(extendDefinition, {
      context: context(),
      hasStoredValues: true,
      storedValues: stored,
      lastGenerationSpec: undefined,
    });
    assert.deepEqual(next.video, { assetId: "user-asset" });
  });

  it("hydrates a prior-generation video before applying the seed", () => {
    const next = resolveFeatureSeedValues(extendDefinition, {
      context: context(),
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: {
        params: {
          prompt: "from history",
          model: "ltx-2.5-fast",
          duration: 6,
          mode: "start",
        },
        inputs: { video: { assetId: "history-video" } },
      },
    });
    assert.deepEqual(next.video, { assetId: "history-video" });
    assert.equal(next.prompt, "from history");
  });

  it("resetValues picks the Settings-matching offering and clears video", () => {
    const reset = extendDefinition.resetValues?.(
      context({ specs: withExtendCaps(specFixtureBothOfferings("LTX 2.3 Fast")) }),
    );
    assert.equal(reset?.model, "ltx-2.3-fast");
    assert.equal(reset?.video, null);
    assert.equal(reset?.prompt, EXTEND_EXAMPLE_PROMPT);
  });

  it("is unavailable when no downloaded offering advertises extend", () => {
    assert.equal(
      extendDefinition.isUnavailable?.(values(), context({ specs: specFixture() })),
      true,
    );
    assert.equal(hasAdvertisedExtend(specFixture()), false);
    assert.equal(
      extendDefinition.isUnavailable?.(values(), context()),
      false,
    );
  });

  it("requires a video whose length can be read", () => {
    assert.equal(extendDefinition.isReady?.(values(), context()), true);
    assert.equal(
      extendDefinition.isReady?.(values({ video: null }), context()),
      false,
    );
    assert.equal(
      extendDefinition.isReady?.(
        values(),
        context({ videoDurationSeconds: null }),
      ),
      false,
    );
  });

  it("sends offering ids, duration, mode, and the video asset id", () => {
    const body = toCreateBody(values({ duration: 8, mode: "start" }));
    assert.deepEqual(body, {
      contract_version: 1,
      inputs: { video: { assetId: "video-1" } },
      params: {
        prompt: EXTEND_EXAMPLE_PROMPT,
        model: "ltx-2.5-fast",
        duration: 8,
        mode: "start",
        promptProvenance: "typed",
      },
    });
  });

  it("rounds duration to one decimal on create", () => {
    const body = toCreateBody(
      values({ duration: 4.724674999999999995 as ExtendValues["duration"] }),
    );
    assert.equal(body.params.duration, 4.7);
  });

  it("restores a generation spec without inventing filesystem paths", () => {
    const restored = fromGeneration(
      {
        params: {
          prompt: "continue",
          model: "ltx-2.5-fast",
          duration: 6,
          mode: "start",
        },
        inputs: { video: { assetId: "clip-9" } },
      },
      context(),
    );
    assert.deepEqual(restored, {
      video: { assetId: "clip-9" },
      prompt: "continue",
      model: "ltx-2.5-fast",
      resolution: "original",
      mode: "start",
      duration: 6,
    });
  });

  it("restores a {width, height} target as its tier", () => {
    const restored = fromGeneration(
      {
        params: {
          prompt: "continue",
          model: "ltx-2.5-fast",
          duration: 6,
          mode: "start",
          resolution: { width: 1280, height: 720 },
        },
        inputs: { video: { assetId: "clip-9" } },
      },
      context(),
    );
    assert.equal(restored?.resolution, "720p");
  });

  it("offers every IC-LoRA cell up to Original for a 720p clip", () => {
    const form = extendDefinition.form(
      values(),
      context({ videoWidth: 1280, videoHeight: 720 }),
    );
    const resolution = form.fields.find((field) => field.id === "resolution");
    assert.equal(resolution?.kind, "options");
    if (resolution?.kind !== "options") return;
    assert.deepEqual(
      resolution.options.map((option) => option.label),
      ["270p", "360p", "540p", "720p (Original)"],
    );
  });

  it("warns about slow runs at 720p and higher on a low performance machine only", () => {
    const warned = (lowPerformanceMachine: boolean) => {
      const base = context({ videoWidth: 1920, videoHeight: 1080 });
      const form = extendDefinition.form(values(), {
        ...base,
        specs: base.specs && {
          ...base.specs,
          low_performance_machine: lowPerformanceMachine,
        },
      });
      const field = form.fields.find((item) => item.id === "resolution");
      return field?.kind === "options"
        ? field.options.filter((option) => option.warning != null).map((option) => option.value)
        : [];
    };
    assert.deepEqual(warned(true), ["720p", "original"]);
    assert.deepEqual(warned(false), []);
  });

  it("sends the 540p tier for a 720p clip", () => {
    const body = toCreateBody(values({ resolution: "540p" }), context({
      videoWidth: 1280,
      videoHeight: 720,
    }));
    assert.deepEqual(body.params.resolution, { width: 960, height: 540 });
  });
});
