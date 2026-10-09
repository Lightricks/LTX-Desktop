import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { VideoGenerationModelSpecsResponse } from "../../../../lib/video-generation-model-specs.ts";
import { resolveFeatureSeedValues } from "../types.ts";
import { specFixture, specFixtureBothOfferings } from "./specFixture.ts";
import {
  RETAKE_DEFAULTS,
  RETAKE_EXAMPLE_DURATION_SECONDS,
  RETAKE_EXAMPLE_PROMPT,
  RETAKE_EXAMPLE_START_SECONDS,
  RETAKE_MAX_DURATION_SECONDS,
  RETAKE_MIN_DURATION_SECONDS,
  RETAKE_RANGE_LABEL,
  RETAKE_VIDEO_RANGE,
  fromGeneration,
  hasAdvertisedRetake,
  retakeDefinition,
  toCreateBody,
  validateRetake,
  type RetakeContext,
  type RetakeValues,
} from "./retake.ts";

const videoAsset = { assetId: "video-1" };

const RETAKE_CAPS = {
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

function withRetakeCaps(
  specs: VideoGenerationModelSpecsResponse,
): VideoGenerationModelSpecsResponse {
  return {
    ...specs,
    downloaded_local_models: specs.downloaded_local_models.map((item) => ({
      ...item,
      spec: { ...item.spec, capabilities: { ...RETAKE_CAPS } },
    })),
  };
}

const seedAsset = { assetId: "seed-video" };

function context(overrides: Partial<RetakeContext> = {}): RetakeContext {
  return {
    specs: withRetakeCaps(specFixture()),
    seed: seedAsset,
    videoDurationSeconds: 8,
    videoDurationPending: false,
    videoWidth: null,
    videoHeight: null,
    videoFps: null,
    ...overrides,
  };
}

function values(overrides: Partial<RetakeValues> = {}): RetakeValues {
  return {
    ...RETAKE_DEFAULTS,
    video: videoAsset,
    ...overrides,
  };
}

describe("retake definition", () => {
  it("does not cap the source video length", () => {
    const form = retakeDefinition.form(values(), context());
    const video = form.fields.find((field) => field.id === "video");
    assert.equal(video?.kind, "video-asset");
    if (video?.kind !== "video-asset") return;
    assert.equal(video.maxDurationSeconds, undefined);
  });

  it("clamps the selection into a trimmed source", () => {
    const normalized = retakeDefinition.normalize?.(
      values({ startTime: 50, duration: 20 }),
      context({ videoDurationSeconds: 10 }),
    );
    assert.equal(normalized?.startTime, 10 - RETAKE_MIN_DURATION_SECONDS);
    assert.equal(normalized?.duration, RETAKE_MIN_DURATION_SECONDS);
  });

  it("caps the selection at the retake maximum on a long source", () => {
    const normalized = retakeDefinition.normalize?.(
      values({ startTime: 5, duration: 40 }),
      context({ videoDurationSeconds: 60 }),
    );
    assert.equal(normalized?.startTime, 5);
    assert.equal(normalized?.duration, RETAKE_MAX_DURATION_SECONDS);
    assert.equal(RETAKE_VIDEO_RANGE.maxDuration, RETAKE_MAX_DURATION_SECONDS);
  });

  it("caps a restored selection before the source length is known", () => {
    const normalized = retakeDefinition.normalize?.(
      values({ duration: 25 }),
      context({ videoDurationSeconds: null }),
    );
    assert.equal(normalized?.duration, RETAKE_MAX_DURATION_SECONDS);
  });

  it("seeds offering from the Settings-matching download", () => {
    const seeded = resolveFeatureSeedValues(retakeDefinition, {
      context: context({ specs: withRetakeCaps(specFixtureBothOfferings()) }),
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: undefined,
    });
    assert.equal(seeded.model, "ltx-2.3-fast");
    assert.equal(seeded.prompt, RETAKE_EXAMPLE_PROMPT);
    assert.equal(seeded.startTime, RETAKE_EXAMPLE_START_SECONDS);
    assert.equal(seeded.duration, RETAKE_EXAMPLE_DURATION_SECONDS);
    assert.deepEqual(seeded.video, seedAsset);
  });

  it("keeps the 5s-to-end example range before the clip length is known", () => {
    const seeded = resolveFeatureSeedValues(retakeDefinition, {
      context: context({
        videoDurationSeconds: null,
        videoDurationPending: false,
        videoWidth: null,
        videoHeight: null,
        videoFps: null,
      }),
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: undefined,
    });
    assert.equal(seeded.startTime, RETAKE_EXAMPLE_START_SECONDS);
    assert.equal(seeded.duration, RETAKE_EXAMPLE_DURATION_SECONDS);
  });

  it("seeds Video from initialValues when the form has never been persisted", () => {
    const seeded = resolveFeatureSeedValues(retakeDefinition, {
      context: context(),
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: undefined,
    });
    assert.deepEqual(seeded.video, seedAsset);
  });

  it("does not reinsert the seed after an explicitly persisted clear", () => {
    const cleared = values({ video: null, prompt: "kept" });
    const next = resolveFeatureSeedValues(retakeDefinition, {
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
    const next = resolveFeatureSeedValues(retakeDefinition, {
      context: context(),
      hasStoredValues: true,
      storedValues: stored,
      lastGenerationSpec: undefined,
    });
    assert.deepEqual(next.video, { assetId: "user-asset" });
  });

  it("hydrates a prior-generation video before applying the seed", () => {
    const next = resolveFeatureSeedValues(retakeDefinition, {
      context: context(),
      hasStoredValues: false,
      storedValues: undefined,
      lastGenerationSpec: {
        params: {
          prompt: "from history",
          model: "ltx-2.5-fast",
          startTime: 0,
          duration: 2,
          mode: "replace_video",
        },
        inputs: { video: { assetId: "history-video" } },
      },
    });
    assert.deepEqual(next.video, { assetId: "history-video" });
    assert.equal(next.prompt, "from history");
  });

  it("resetValues picks the Settings-matching offering and clears video", () => {
    const reset = retakeDefinition.resetValues?.(
      context({ specs: withRetakeCaps(specFixtureBothOfferings("LTX 2.3 Fast")) }),
    );
    assert.equal(reset?.model, "ltx-2.3-fast");
    assert.equal(reset?.video, null);
    assert.equal(reset?.prompt, RETAKE_EXAMPLE_PROMPT);
    assert.equal(reset?.startTime, RETAKE_EXAMPLE_START_SECONDS);
    assert.equal(reset?.duration, RETAKE_EXAMPLE_DURATION_SECONDS);
  });

  it("keeps the edit range off the shared video field type", () => {
    const form = retakeDefinition.form(values(), context());
    assert.deepEqual(
      form.fields.map((field) => field.id),
      ["video", "prompt", "model"],
    );
    const video = form.fields[0];
    assert.equal(video?.kind, "video-asset");
    if (video?.kind !== "video-asset") return;
    assert.equal("inlineRange" in video, false);
    assert.deepEqual(RETAKE_VIDEO_RANGE, {
      startDataKey: "startTime",
      durationDataKey: "duration",
      minDuration: RETAKE_MIN_DURATION_SECONDS,
      maxDuration: RETAKE_MAX_DURATION_SECONDS,
      label: RETAKE_RANGE_LABEL,
    });
  });

  it("attaches an over-long region error to the video field", () => {
    const issues = validateRetake(
      values({ startTime: 0, duration: 10 }),
      context({ videoDurationSeconds: 5 }),
    );
    assert.equal(issues[0]?.id, "retake-selection-out-of-range");
    assert.equal(issues[0]?.fieldId, "video");
    assert.equal(issues[0]?.alwaysRevealed, true);
  });

  it("is unavailable when no downloaded offering advertises retake", () => {
    assert.equal(
      retakeDefinition.isUnavailable?.(values(), context({ specs: specFixture() })),
      true,
    );
    assert.equal(hasAdvertisedRetake(specFixture()), false);
    assert.equal(
      retakeDefinition.isUnavailable?.(values(), context()),
      false,
    );
  });

  it("requires a video long enough for the 2s region", () => {
    assert.equal(retakeDefinition.isReady?.(values(), context()), true);
    assert.equal(
      retakeDefinition.isReady?.(values({ video: null }), context()),
      false,
    );
    assert.equal(
      retakeDefinition.isReady?.(values(), context({ videoDurationSeconds: 1.5 })),
      false,
    );
    const issues = validateRetake(
      values(),
      context({ videoDurationSeconds: 1.5 }),
    );
    assert.equal(issues[0]?.id, "video-shorter-than-minimum");
    assert.equal(
      issues[0]?.message,
      `This video is shorter than ${RETAKE_MIN_DURATION_SECONDS}s. Replace it to continue.`,
    );
    assert.equal(issues[0]?.alwaysRevealed, true);
  });

  it("accepts a source longer than 60s when the selection is within 10s", () => {
    assert.deepEqual(
      validateRetake(
        values({ startTime: 0, duration: 5 }),
        context({ videoDurationSeconds: 75 }),
      ),
      [],
    );
  });

  it("sends offering ids, region, and the video asset id", () => {
    const body = toCreateBody(values({ startTime: 1, duration: 3 }));
    assert.deepEqual(body, {
      contract_version: 1,
      inputs: { video: { assetId: "video-1" } },
      params: {
        prompt: RETAKE_EXAMPLE_PROMPT,
        model: "ltx-2.5-fast",
        startTime: 1,
        duration: 3,
        mode: "replace_audio_and_video",
        promptProvenance: "typed",
      },
    });
  });

  it("rounds duration to one decimal on create", () => {
    const body = toCreateBody(
      values({ startTime: 0, duration: 4.724674999999999995 }),
    );
    assert.equal(body.params.duration, 4.7);
  });

  it("restores a generation spec without inventing filesystem paths", () => {
    const restored = fromGeneration(
      {
        params: {
          prompt: "turn",
          model: "ltx-2.5-fast",
          startTime: 0.5,
          duration: 2,
          mode: "replace_audio",
        },
        inputs: { video: { assetId: "clip-9" } },
      },
      context(),
    );
    assert.deepEqual(restored, {
      video: { assetId: "clip-9" },
      prompt: "turn",
      model: "ltx-2.5-fast",
      resolution: "original",
      startTime: 0.5,
      duration: 2,
    });
  });

  it("restores a {width, height} target as its tier", () => {
    const restored = fromGeneration(
      {
        params: {
          prompt: "turn",
          model: "ltx-2.5-fast",
          startTime: 0.5,
          duration: 2,
          mode: "replace_audio",
          resolution: { width: 1280, height: 720 },
        },
        inputs: { video: { assetId: "clip-9" } },
      },
      context(),
    );
    assert.equal(restored?.resolution, "720p");
    // A ÷32 snap of 1080p is 1056, which is still the 1080p cell, not Original.
    const snapped = fromGeneration(
      {
        params: {
          prompt: "turn",
          model: "ltx-2.5-fast",
          startTime: 0.5,
          duration: 2,
          mode: "replace_audio",
          resolution: { width: 1920, height: 1056 },
        },
        inputs: { video: { assetId: "clip-9" } },
      },
      context(),
    );
    assert.equal(snapped?.resolution, "1080p");
  });

  it("warns about slow runs at 720p and higher on a low performance machine only", () => {
    const warned = (
      videoWidth: number,
      videoHeight: number,
      lowPerformanceMachine: boolean,
    ) => {
      const base = context({ videoWidth, videoHeight });
      const form = retakeDefinition.form(values(), {
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
    assert.deepEqual(warned(1920, 1080, true), ["720p", "original"]);
    assert.deepEqual(warned(3840, 2160, true), ["720p", "original"]);
    assert.deepEqual(warned(1280, 720, true), ["original"]);
    assert.deepEqual(warned(1920, 1080, false), []);
  });

  function resolutionLabels(videoWidth: number, videoHeight: number): string[] {
    const form = retakeDefinition.form(values(), context({ videoWidth, videoHeight }));
    const resolution = form.fields.find((field) => field.id === "resolution");
    assert.equal(resolution?.kind, "options");
    if (resolution?.kind !== "options") return [];
    return resolution.options.map((option) => option.label);
  }

  it("offers every IC-LoRA cell up to Original, and never a cell above the clip", () => {
    assert.deepEqual(resolutionLabels(1920, 1080), [
      "270p",
      "360p",
      "540p",
      "720p",
      "1080p (Original)",
    ]);
    // The local cap is 1080p, so a 4K clip tags 1080p as Original.
    assert.deepEqual(resolutionLabels(3840, 2160), [
      "270p",
      "360p",
      "540p",
      "720p",
      "1080p (Original)",
    ]);
    assert.deepEqual(resolutionLabels(1024, 576), ["270p", "360p", "540p", "576p (Original)"]);
  });

  it("a clip between two cells shows its real size as Original", () => {
    assert.deepEqual(resolutionLabels(854, 480), ["270p", "360p", "480p (Original)"]);
    assert.deepEqual(resolutionLabels(1920, 800), ["270p", "360p", "540p", "720p", "800p (Original)"]);
    assert.deepEqual(resolutionLabels(480, 854), ["270p", "360p", "480p (Original)"]);
  });

  it("a size request is dropped when the cell is not below the clip", () => {
    const body = toCreateBody(values({ resolution: "720p" }), context({
      videoWidth: 640,
      videoHeight: 360,
    }));
    assert.equal(body.params.resolution, undefined);
  });

  it("a clip under 540p is labelled Original at its own size and offers no cell above it", () => {
    assert.deepEqual(resolutionLabels(640, 360), ["270p", "360p (Original)"]);
  });

  it("a stored resolution above the clip falls back to original", () => {
    const normalized = retakeDefinition.normalize?.(
      values({ resolution: "540p" }),
      context({ videoWidth: 640, videoHeight: 360 }),
    );
    assert.equal(normalized?.resolution, "original");
  });

  it("sends the 720p tier and omits original", () => {
    const chosen = toCreateBody(values({ resolution: "720p" }), context({
      videoWidth: 1920,
      videoHeight: 1080,
    }));
    assert.deepEqual(chosen.params.resolution, { width: 1280, height: 720 });

    const original = toCreateBody(values({ resolution: "original" }), context({
      videoWidth: 1920,
      videoHeight: 1080,
    }));
    assert.equal(original.params.resolution, undefined);
  });
});
