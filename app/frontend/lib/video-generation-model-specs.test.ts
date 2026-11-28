import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  cheapestDroppedCellMessage,
  type ChunkedFastCell,
  resolveVideoGenerationOptions,
  withoutBetCells,
  type DroppedFastAspectCell,
  type DroppedFastCell,
  type VideoGenerationModelSpecItem,
  type VideoGenerationModelSpecsResponse,
  type WarnedFastCell,
} from "./video-generation-model-specs.ts";

type KeptGrid = Record<string, { fps_to_durations: Record<string, number[]> }>;

function specItem(
  kept: KeptGrid,
  dropped: DroppedFastCell[],
  a2v: KeptGrid = {},
  aspectDropped: DroppedFastAspectCell[] = [],
): VideoGenerationModelSpecItem {
  return {
    pipeline: "fast",
    spec: {
      display_name: "LTX 2.5 Fast",
      supported_resolutions_durations:
        kept as VideoGenerationModelSpecItem["spec"]["supported_resolutions_durations"],
      a2v_supported_resolutions_durations:
        a2v as VideoGenerationModelSpecItem["spec"]["a2v_supported_resolutions_durations"],
      capabilities: null,
      dropped_cells: dropped,
      dropped_aspect_cells: aspectDropped,
    },
  };
}

function dropped(
  resolution: DroppedFastCell["resolution"],
  fps: DroppedFastCell["fps"],
  duration: DroppedFastCell["duration"],
  message: string,
  audio = false,
): DroppedFastCell {
  return { resolution, fps, duration, reason: "stream_over", message, audio };
}

function aspectDropped(
  resolution: DroppedFastAspectCell["resolution"],
  fps: DroppedFastAspectCell["fps"],
  duration: DroppedFastAspectCell["duration"],
  aspectRatio: DroppedFastAspectCell["aspectRatio"],
  message: string,
  audio = false,
): DroppedFastAspectCell {
  return { resolution, fps, duration, aspectRatio, reason: "stream_over", message, audio };
}

describe("resolveVideoGenerationOptions disabled options", () => {
  it("returns exact duration messages and cheapest field messages, kept-only selections", () => {
    const item = specItem(
      {
        "540p": { fps_to_durations: { "24": [5, 10, 20] } },
        "720p": { fps_to_durations: { "24": [5, 10] } },
      },
      [
        dropped("720p", 24, 20, "needs 15.8"),
        dropped("1080p", 24, 5, "needs 18.3"),
        dropped("1080p", 24, 10, "needs 18.3+"),
      ],
    );
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "720p", fps: 24 },
      modelSpecs: [item],
    });

    assert.deepEqual(resolved.durationOptions, [5, 10]);
    assert.deepEqual(resolved.disabledDurationOptions, [
      { value: 20, message: "needs 15.8" },
    ]);
    assert.deepEqual(resolved.disabledResolutionOptions, [
      { value: "1080p", message: "needs 18.3" },
    ]);
    assert.deepEqual(resolved.disabledFpsOptions, []);
    assert.equal(resolved.selectedDuration, 5);
    assert.equal(resolved.hasCompatibleOptions, true);
  });

  it("groups dropped-only fps with the cheapest message", () => {
    const item = specItem(
      { "720p": { fps_to_durations: { "24": [5] } } },
      [
        dropped("720p", 48, 5, "needs f48"),
        dropped("720p", 48, 10, "needs f48+"),
      ],
    );
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "720p", fps: 24 },
      modelSpecs: [item],
    });

    assert.deepEqual(resolved.fpsOptions, [24]);
    assert.deepEqual(resolved.disabledFpsOptions, [
      { value: 48, message: "needs f48" },
    ]);
  });

  it("filters the dropped matrix by audio both ways", () => {
    const item = specItem(
      { "540p": { fps_to_durations: { "24": [5] } } },
      [
        dropped("540p", 24, 10, "ram-msg", true),
        dropped("540p", 24, 20, "plain-msg", false),
      ],
      { "540p": { fps_to_durations: { "24": [5] } } },
    );
    const audio = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "540p", fps: 24 },
      modelSpecs: [item],
      hasAudio: true,
    });
    const plain = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "540p", fps: 24 },
      modelSpecs: [item],
      hasAudio: false,
    });

    assert.deepEqual(audio.disabledDurationOptions, [
      { value: 10, message: "ram-msg" },
    ]);
    assert.deepEqual(plain.disabledDurationOptions, [
      { value: 20, message: "plain-msg" },
    ]);
  });

  it("falls back to first-row disabled resolutions when no model is compatible", () => {
    const item = specItem({}, [
      dropped("720p", 24, 5, "m720"),
      dropped("1080p", 24, 5, "m1080"),
    ]);
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "540p", fps: 24 },
      modelSpecs: [item],
    });

    assert.equal(resolved.selectedModel, null);
    assert.deepEqual(resolved.disabledResolutionOptions, [
      { value: "720p", message: "m720" },
      { value: "1080p", message: "m1080" },
    ]);
    assert.deepEqual(resolved.disabledFpsOptions, []);
    assert.deepEqual(resolved.disabledDurationOptions, []);
  });

  it("does not leak a refused shape to sibling cells", () => {
    const item = specItem(
      { "540p": { fps_to_durations: { "24": [5, 20] } } },
      [],
      {},
      [aspectDropped("540p", 24, 20, "21:9", "needs 17.6")],
    );
    const refused = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 20, videoResolution: "540p", fps: 24 },
      modelSpecs: [item],
    });
    const sibling = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "540p", fps: 24 },
      modelSpecs: [item],
    });

    assert.deepEqual(refused.disabledAspectRatioOptions, [
      { value: "21:9", message: "needs 17.6" },
    ]);
    assert.deepEqual(sibling.disabledAspectRatioOptions, []);
  });

  it("does not leak aspect refusals across audio matrices", () => {
    const item = specItem(
      { "540p": { fps_to_durations: { "24": [5] } } },
      [],
      { "540p": { fps_to_durations: { "24": [5] } } },
      [aspectDropped("540p", 24, 5, "21:9", "a2v-msg", true)],
    );
    const audio = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "540p", fps: 24 },
      modelSpecs: [item],
      hasAudio: true,
    });
    const plain = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "540p", fps: 24 },
      modelSpecs: [item],
      hasAudio: false,
    });

    assert.deepEqual(audio.disabledAspectRatioOptions, [
      { value: "21:9", message: "a2v-msg" },
    ]);
    assert.deepEqual(plain.disabledAspectRatioOptions, []);
  });
});

describe("resolveVideoGenerationOptions warnings", () => {
  it("marks a kept tight cell without removing it from the offered durations", () => {
    const item = specItem(
      { "720p": { fps_to_durations: { "24": [5] } } },
      [],
    );
    const warned: WarnedFastCell = {
      resolution: "720p",
      fps: 24,
      duration: 5,
      tier: "tight",
      message: "tight fit",
      audio: false,
    };
    item.spec.warned_cells = [warned];
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "720p", fps: 24 },
      modelSpecs: [item],
    });
    assert.deepEqual(resolved.durationOptions, [5]);
    assert.deepEqual(resolved.warnedDurationOptions, [
      { value: 5, message: "tight fit", tier: "tight" },
    ]);
  });
});

describe("resolveVideoGenerationOptions chunked cells", () => {
  const chunked = (
    resolution: ChunkedFastCell["resolution"],
    fps: ChunkedFastCell["fps"],
    duration: ChunkedFastCell["duration"],
    windows: number,
    audio = false,
  ): ChunkedFastCell => ({ resolution, fps, duration, windows, window_frames: 241, audio });

  function item(cells: ChunkedFastCell[]): VideoGenerationModelSpecItem {
    const base = specItem(
      {
        "720p": { fps_to_durations: { "24": [5, 10, 20], "48": [5, 10] } },
        "1080p": { fps_to_durations: { "24": [5, 10, 20], "48": [5, 10] } },
      },
      [],
    );
    base.spec.chunked_cells = cells;
    return base;
  }

  it("lists the cells that run in windows on the duration, fps and resolution pickers", () => {
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 20, videoResolution: "1080p", fps: 24 },
      modelSpecs: [item([chunked("1080p", 24, 20, 3), chunked("1080p", 48, 10, 2)])],
    });
    assert.deepEqual(resolved.chunkedDurationOptions, [{ value: 20, windows: 3, windowFrames: 241 }]);
    assert.deepEqual(resolved.chunkedResolutionOptions, [{ value: "1080p", windows: 3, windowFrames: 241 }]);
    // The fps picker reads the selected duration: 24 fps is the selected cell, 48 fps at 20 s is not offered.
    assert.deepEqual(resolved.chunkedFpsOptions, [{ value: 24, windows: 3, windowFrames: 241 }]);
  });

  it("judges the fps picker at the selected duration", () => {
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 10, videoResolution: "1080p", fps: 24 },
      modelSpecs: [item([chunked("1080p", 48, 10, 2)])],
    });
    assert.deepEqual(resolved.chunkedFpsOptions, [{ value: 48, windows: 2, windowFrames: 241 }]);
    assert.deepEqual(resolved.chunkedDurationOptions, []);
  });

  it("does not mix the A2V grid into the plain one", () => {
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 20, videoResolution: "1080p", fps: 24 },
      modelSpecs: [item([chunked("1080p", 24, 20, 3, true)])],
    });
    assert.deepEqual(resolved.chunkedDurationOptions, []);
  });

  it("names the resolutions that offer the selected cell and run it in one pass", () => {
    const base = specItem(
      {
        "720p": { fps_to_durations: { "24": [5, 10, 20] } },
        "1080p": { fps_to_durations: { "24": [5, 10, 20] } },
        // 4K stops at 10 s: choosing it from 20 s changes the duration, so its cell is unknown.
        "2160p": { fps_to_durations: { "24": [5, 10] } },
      },
      [],
    );
    base.spec.chunked_cells = [chunked("1080p", 24, 20, 3)];
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 20, videoResolution: "1080p", fps: 24 },
      modelSpecs: [base],
    });
    assert.deepEqual(resolved.singlePassResolutionOptions, ["720p"]);
    assert.deepEqual(resolved.chunkedResolutionOptions, [{ value: "1080p", windows: 3, windowFrames: 241 }]);
  });

  it("leaves a cell that runs in one pass out", () => {
    const resolved = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 5, videoResolution: "720p", fps: 24 },
      modelSpecs: [item([chunked("1080p", 24, 20, 3)])],
    });
    assert.deepEqual(resolved.chunkedDurationOptions, []);
    assert.deepEqual(resolved.chunkedResolutionOptions, []);
  });
});

describe("cheapestDroppedCellMessage", () => {
  it("orders one way everywhere: resolution, then duration, then fps", () => {
    // Home unioned fps-first while GenSpace scoped duration-first, so the
    // same withheld option tooltiped different refusals per surface.
    assert.equal(
      cheapestDroppedCellMessage([
        dropped("720p", 48, 5, "720p-dur5"),
        dropped("540p", 50, 10, "540p-dur10-fps50"),
        dropped("540p", 48, 10, "540p-dur10-fps48"),
      ]),
      "540p-dur10-fps48",
    );
  });
});

describe("per-aspect spec resolution and bet stripping", () => {
  const aspects = ["21:9", "16:9", "3:2", "4:3", "1:1", "4:5", "9:16"] as const;

  it("test_resolve_video_generation_options_switches_duration_grid_and_disabled_aspects_by_selected_aspect", () => {
    const item = specItem(
      { "1080p": { fps_to_durations: { "24": [5] }, aspect_ratios: [...aspects] } as never },
      [dropped("1080p", 24, 10, "16:9 needs 22.7 GiB")],
    );
    item.spec.aspect_specs = {
      "16:9": {
        supported_resolutions_durations: {
          "1080p": { fps_to_durations: { "24": [5] }, aspect_ratios: [...aspects] },
        },
        dropped_cells: [dropped("1080p", 24, 10, "16:9 needs 22.7 GiB")],
      },
      "1:1": {
        supported_resolutions_durations: {
          "1080p": { fps_to_durations: { "24": [5, 10] }, aspect_ratios: [...aspects] },
        },
        dropped_cells: [],
      },
    };

    const at16x9 = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 10, videoResolution: "1080p", fps: 24, aspectRatio: "16:9" },
      modelSpecs: [item],
    });
    assert.deepEqual(at16x9.durationOptions, [5]);
    assert.deepEqual(at16x9.disabledDurationOptions, [{ value: 10, message: "16:9 needs 22.7 GiB" }]);

    const at1x1 = resolveVideoGenerationOptions({
      settings: { model: "fast", duration: 10, videoResolution: "1080p", fps: 24, aspectRatio: "1:1" },
      modelSpecs: [item],
    });
    assert.deepEqual(at1x1.durationOptions, [5, 10]);
    assert.equal(at1x1.selectedDuration, 10);
    assert.deepEqual(at1x1.disabledAspectRatioOptions, [
      { value: "16:9", message: "16:9 needs 22.7 GiB" },
    ]);
  });

  it("test_strip_bet_cells_strips_over_and_unmeasured_cells_inside_per_aspect_specs_in_cautious_mode", () => {
    const item = specItem(
      { "1080p": { fps_to_durations: { "24": [5, 10] }, aspect_ratios: [...aspects] } as never },
      [],
    );
    const overCell: WarnedFastCell = {
      resolution: "1080p",
      fps: 24,
      duration: 10,
      tier: "over",
      message: "1:1 10s over budget",
      audio: false,
    };
    item.spec.aspect_specs = {
      "1:1": {
        supported_resolutions_durations: {
          "1080p": { fps_to_durations: { "24": [5, 10] }, aspect_ratios: [...aspects] },
        },
        warned_cells: [overCell],
        dropped_cells: [],
      },
    };
    const response: VideoGenerationModelSpecsResponse = {
      api_models: [],
      local_models: [item],
      downloaded_local_models: [{ model: "ltx-2.5-fast", spec: item.spec }],
    };

    const stripped = withoutBetCells(response);
    const aspectSpec = stripped.local_models[0]?.spec.aspect_specs?.["1:1"];
    assert.deepEqual(
      aspectSpec?.supported_resolutions_durations["1080p"]?.fps_to_durations["24"],
      [5],
    );
    assert.equal(aspectSpec?.warned_cells?.length, 0);
    assert.equal(aspectSpec?.dropped_cells?.[0]?.duration, 10);
    assert.equal(aspectSpec?.dropped_cells?.[0]?.message, "1:1 10s over budget");
  });
});

