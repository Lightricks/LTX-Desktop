import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type {
  DroppedFastAspectCell,
  DroppedFastCell,
  VideoGenerationModelSpecItem,
  VideoGenerationModelSpecsResponse,
} from "../../lib/video-generation-model-specs.ts";
import { specFixture } from "../screens/Feature/definitions/specFixture.ts";
import { TEXT_TO_VIDEO_DEFAULTS } from "../screens/Feature/definitions/textToVideo.ts";
import {
  clampVideoFields,
  collectModelDisabledFpsOptions,
  isLocalVideoSelectionReady,
  retargetVideoFields,
} from "./videoFieldPolicy.ts";

type VideoValues = typeof TEXT_TO_VIDEO_DEFAULTS;

function values(overrides: Partial<VideoValues> = {}): VideoValues {
  return { ...TEXT_TO_VIDEO_DEFAULTS, ...overrides };
}

describe("clampVideoFields", () => {
  const specs = specFixture();

  it("keeps the current value when it is still legal", () => {
    const current = values({
      model: "ltx-2.5-fast",
      resolution: "720p",
      fps: 24,
      duration: 8,
    });
    assert.deepEqual(
      clampVideoFields(current, specs, TEXT_TO_VIDEO_DEFAULTS),
      current,
    );
  });

  it("clamps an id the downloaded catalog doesn't expose down to the default offering", () => {
    const clamped = clampVideoFields(
      values({ model: "pro" }),
      specs,
      TEXT_TO_VIDEO_DEFAULTS,
    );
    assert.equal(clamped.model, "ltx-2.5-fast");
  });

  it("falls back to the field default when current is illegal but default is in the list", () => {
    const clamped = clampVideoFields(
      values({ resolution: "2160p" }),
      specs,
      TEXT_TO_VIDEO_DEFAULTS,
    );
    assert.equal(clamped.resolution, "540p");
  });

  it("falls back to the first remaining option when the default is not in the list", () => {
    const clamped = clampVideoFields(
      values({ resolution: "2160p" }),
      specs,
      { ...TEXT_TO_VIDEO_DEFAULTS, resolution: "1440p" },
    );
    assert.equal(clamped.resolution, "540p");
  });

  it("keeps 2s through 5s when the spec advertises them", () => {
    const fromFive = clampVideoFields(
      values({ duration: 5 }),
      specs,
      TEXT_TO_VIDEO_DEFAULTS,
    );
    assert.equal(fromFive.duration, 5);

    const fromTwo = clampVideoFields(
      values({ duration: 2 }),
      specs,
      TEXT_TO_VIDEO_DEFAULTS,
    );
    assert.equal(fromTwo.duration, 2);

    const from1080pFive = clampVideoFields(
      values({ resolution: "1080p", duration: 5 }),
      specs,
      TEXT_TO_VIDEO_DEFAULTS,
    );
    assert.equal(from1080pFive.resolution, "1080p");
    assert.equal(from1080pFive.duration, 5);
  });

  it("preserves Auto aspect ratio instead of replacing it with 16:9", () => {
    const clamped = clampVideoFields(
      { ...values(), aspectRatio: "auto" },
      specs,
      TEXT_TO_VIDEO_DEFAULTS,
    );
    assert.equal(clamped.aspectRatio, "auto");
  });

  it("picks the default duration when it is in the list but not first", () => {
    const clamped = clampVideoFields(
      values({ duration: 12 }),
      specs,
      { ...TEXT_TO_VIDEO_DEFAULTS, duration: 10 },
    );
    assert.equal(clamped.duration, 10);
  });
});

describe("retargetVideoFields", () => {
  const specs = specFixture();

  it("switches resolution so a duration that is illegal here can stick", () => {
    // 1080p can't run 8s locally; retargeting to preserve duration moves to the
    // first resolution that supports 8s at 24fps (540p).
    const next = retargetVideoFields(
      values({
        model: "ltx-2.5-fast",
        resolution: "1080p",
        fps: 24,
        duration: 8,
      }),
      specs,
      "duration",
    );
    assert.equal(next.duration, 8);
    assert.equal(next.resolution, "540p");
    assert.equal(next.fps, 24);
  });
});

function unionItem(
  kept: Record<string, { fps_to_durations: Record<string, number[]> }>,
  dropped: DroppedFastCell[],
): VideoGenerationModelSpecItem {
  return {
    pipeline: "fast",
    spec: {
      display_name: "LTX 2.5 Fast",
      supported_resolutions_durations:
        kept as VideoGenerationModelSpecItem["spec"]["supported_resolutions_durations"],
      a2v_supported_resolutions_durations: {},
      capabilities: null,
      dropped_cells: dropped,
    },
  };
}

function unionDropped(
  resolution: DroppedFastCell["resolution"],
  fps: DroppedFastCell["fps"],
  duration: DroppedFastCell["duration"],
  message: string,
): DroppedFastCell {
  return { resolution, fps, duration, reason: "stream_over", message, audio: false };
}

describe("collectModelDisabledFpsOptions", () => {
  it("lists fps dropped everywhere with the cheapest message, kept nowhere", () => {
    const item = unionItem({ "540p": { fps_to_durations: { "24": [5] } } }, [
      unionDropped("540p", 48, 5, "f48"),
      unionDropped("540p", 24, 10, "d24"),
    ]);

    assert.deepEqual(collectModelDisabledFpsOptions(item, [24]), [
      { value: 48, message: "f48" },
    ]);
  });

  it("returns empty without specs", () => {
    assert.deepEqual(collectModelDisabledFpsOptions(undefined, []), []);
  });
});

const REFUSED_ASPECTS = ["21:9", "16:9", "3:2", "4:3", "1:1", "4:5", "9:16"] as const;

function specsWithAspectDropped(
  cells: DroppedFastAspectCell[],
): VideoGenerationModelSpecsResponse {
  const base = specFixture();
  return {
    ...base,
    downloaded_local_models: base.downloaded_local_models.map((item) => ({
      ...item,
      spec: {
        ...item.spec,
        supported_resolutions_durations: {
          "540p": {
            fps_to_durations: { "24": [2, 3, 4, 5, 6, 8, 10, 20] },
            aspect_ratios: [...REFUSED_ASPECTS],
          },
        },
        dropped_cells: [],
        dropped_aspect_cells: cells,
      },
    })),
  };
}

function aspectDropped(
  aspectRatio: DroppedFastAspectCell["aspectRatio"],
): DroppedFastAspectCell {
  return {
    resolution: "540p",
    fps: 24,
    duration: 8,
    aspectRatio,
    reason: "stream_over",
    message: `${aspectRatio} cannot run 540p/8s here`,
    audio: false,
  };
}

describe("refused aspects", () => {
  it("clamps off a refused aspect onto the default", () => {
    const clamped = clampVideoFields(
      values({ aspectRatio: "21:9", resolution: "540p", fps: 24, duration: 8 }),
      specsWithAspectDropped([aspectDropped("21:9")]),
      TEXT_TO_VIDEO_DEFAULTS,
    );
    assert.equal(clamped.aspectRatio, "16:9");
  });

  it("keeps a refused aspect when every shape is refused", () => {
    const clamped = clampVideoFields(
      values({ aspectRatio: "21:9", resolution: "540p", fps: 24, duration: 8 }),
      specsWithAspectDropped(REFUSED_ASPECTS.map(aspectDropped)),
      TEXT_TO_VIDEO_DEFAULTS,
    );
    assert.equal(clamped.aspectRatio, "21:9");
  });

  it("reports not ready while a refused aspect is selected", () => {
    assert.equal(
      isLocalVideoSelectionReady(
        values({ aspectRatio: "21:9", resolution: "540p", fps: 24, duration: 8 }),
        specsWithAspectDropped([aspectDropped("21:9")]),
      ),
      false,
    );
    assert.equal(
      isLocalVideoSelectionReady(
        values({ aspectRatio: "16:9", resolution: "540p", fps: 24, duration: 8 }),
        specsWithAspectDropped([aspectDropped("21:9")]),
      ),
      true,
    );
  });
});
