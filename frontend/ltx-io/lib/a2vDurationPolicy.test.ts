import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type {
  VideoGenerationDuration,
  VideoGenerationModelSpecsResponse,
} from "../../lib/video-generation-model-specs.ts";
import { specFixture } from "../screens/Feature/definitions/specFixture.ts";
import {
  a2vEffectiveAudioSeconds,
  a2vLongestCellSeconds,
  a2vNumFramesForAudio,
  advertisedA2vDurationsAt540p24,
  advertisedA2vDurations,
  a2vResolutionsForOffering,
  a2vTrimCapSeconds,
  computeA2vMaxFrames,
  formatA2vDurationBadge,
  hasAdvertisedA2vCell,
  numFramesForAudioDuration,
} from "./a2vDurationPolicy.ts";

const LOCAL_ASPECTS = ["21:9", "16:9", "3:2", "4:3", "1:1", "4:5", "9:16"] as const;

function a2vCell(durations: VideoGenerationDuration[]) {
  return {
    fps_to_durations: { "24": durations },
    aspect_ratios: [...LOCAL_ASPECTS],
  };
}

function a2vSpecs(
  durations: VideoGenerationDuration[],
): VideoGenerationModelSpecsResponse {
  const spec = {
    display_name: "LTX 2.5 Fast",
    supported_resolutions_durations: {},
    a2v_supported_resolutions_durations: {
      "540p": a2vCell(durations),
    },
  };
  return {
    api_models: [],
    local_models: [
      {
        pipeline: "fast",
        spec,
      },
    ],
    downloaded_local_models: [
      {
        model: "ltx-2.5-fast",
        pipeline: "fast",
        spec,
      },
    ],
  };
}

describe("a2vDurationPolicy", () => {
  it("reads advertised A2V durations from the selected offering, not Settings-active", () => {
    const specs: VideoGenerationModelSpecsResponse = {
      api_models: [],
      local_models: [
        {
          pipeline: "fast",
          spec: {
            display_name: "LTX 2.3 Fast",
            supported_resolutions_durations: {},
            a2v_supported_resolutions_durations: {
              "540p": a2vCell([5]),
            },
          },
        },
      ],
      downloaded_local_models: [
        {
          model: "ltx-2.5-fast",
          pipeline: "fast",
          spec: {
            display_name: "LTX 2.5 Fast",
            supported_resolutions_durations: {},
            a2v_supported_resolutions_durations: {
              "540p": a2vCell([5, 6, 8, 10, 20]),
            },
          },
        },
        {
          model: "ltx-2.3-fast",
          pipeline: "fast",
          spec: {
            display_name: "LTX 2.3 Fast",
            supported_resolutions_durations: {},
            a2v_supported_resolutions_durations: {
              "540p": a2vCell([5]),
            },
          },
        },
      ],
      active_offering: "ltx-2.3-fast",
    };
    assert.deepEqual(advertisedA2vDurationsAt540p24(specs, "ltx-2.5-fast"), [
      5, 6, 8, 10, 20,
    ]);
    assert.equal(a2vTrimCapSeconds(specs, "ltx-2.5-fast"), 20);
    assert.deepEqual(advertisedA2vDurationsAt540p24(specs, "ltx-2.3-fast"), [5]);
    assert.equal(a2vTrimCapSeconds(specs, "ltx-2.3-fast"), 5);
    // No offering: Settings-matching downloaded row (2.3).
    assert.deepEqual(advertisedA2vDurationsAt540p24(specs), [5]);
    assert.equal(a2vTrimCapSeconds(specs), 5);
  });

  it("reads advertised A2V durations at 540p/24 from local fast specs", () => {
    assert.deepEqual(advertisedA2vDurationsAt540p24(specFixture()), [
      5, 6, 8, 10, 20,
    ]);
    assert.deepEqual(advertisedA2vDurationsAt540p24(a2vSpecs([5, 6, 8, 10, 20])), [
      5, 6, 8, 10, 20,
    ]);
  });

  it("trim cap is the max advertised duration", () => {
    assert.equal(a2vTrimCapSeconds(specFixture()), 20);
    assert.equal(a2vTrimCapSeconds(null), null);
    assert.equal(a2vTrimCapSeconds(a2vSpecs([5, 6, 8, 10, 20])), 20);
  });

  it("floors 10.563s onto 249 frames under the 20s envelope, not 481", () => {
    assert.equal(numFramesForAudioDuration(10.563, 24, 481), 249);
    assert.equal(a2vNumFramesForAudio(10.563, [6, 8, 10, 20]), 249);
  });

  it("reports over-cap when floored frames exceed the envelope", () => {
    assert.equal(numFramesForAudioDuration(10.563, 24, 241), null);
    assert.equal(a2vNumFramesForAudio(10.563, [10]), null);
  });

  it("floors exact 5s to 113, not the T2V 121 cell", () => {
    assert.equal(numFramesForAudioDuration(5.0, 24, 481), 113);
  });

  it("keeps the 2s minimum on the 8k+1 grid", () => {
    const frames = numFramesForAudioDuration(2.0, 24, 481);
    assert.ok(frames !== null);
    assert.ok(frames >= 9);
    assert.equal((frames - 1) % 8, 0);
  });

  it("rejects under 2s and at/over cap +0.1s", () => {
    assert.equal(a2vNumFramesForAudio(1.0, [6, 8, 10, 20]), null);
    assert.equal(a2vNumFramesForAudio(20.1, [6, 8, 10, 20]), null);
    assert.equal(a2vNumFramesForAudio(10.563, []), null);
    // Within tolerance and frames fit: still valid.
    assert.equal(a2vNumFramesForAudio(20.05, [6, 8, 10, 20]), 481);
  });

  it("computes max frames from the advertised cap", () => {
    assert.equal(computeA2vMaxFrames(10, 24), 241);
    assert.equal(computeA2vMaxFrames(20, 24), 481);
  });

  it("formats derived duration badges from persisted frames", () => {
    assert.equal(formatA2vDurationBadge(249, 24), "10.38s");
    assert.equal(formatA2vDurationBadge(185, 24), "7.71s");
  });

  it("keeps a 5s-only advertised envelope available with a 2s input floor", () => {
    // Backend parity: Fast 540p/24 [5] is a valid envelope; GenSpace's 6s
    // picker floor must not filter A2V cells. Cap is the advertised max.
    assert.deepEqual(advertisedA2vDurationsAt540p24(a2vSpecs([5])), [5]);
    assert.equal(a2vTrimCapSeconds(a2vSpecs([5])), 5);
    assert.equal(computeA2vMaxFrames(5, 24), 121);
    // >=2s and under cap +0.1 may generate.
    assert.equal(a2vNumFramesForAudio(2.0, [5]), 41);
    assert.equal(a2vNumFramesForAudio(4.5, [5]), 105);
    assert.equal(a2vNumFramesForAudio(5.05, [5]), 121);
    // Under 2s or at/over cap +0.1 stays rejected.
    assert.equal(a2vNumFramesForAudio(1.9, [5]), null);
    assert.equal(a2vNumFramesForAudio(5.1, [5]), null);
  });

  it("does not use Settings local_models when nothing is downloaded", () => {
    const specs: VideoGenerationModelSpecsResponse = {
      api_models: [],
      local_models: specFixture().local_models,
      downloaded_local_models: [],
    };
    assert.deepEqual(advertisedA2vDurationsAt540p24(specs), []);
    assert.equal(a2vTrimCapSeconds(specs), null);
    assert.equal(hasAdvertisedA2vCell(specs), false);
  });

  it("returns no cell when the selected offering is not downloaded", () => {
    assert.deepEqual(
      advertisedA2vDurationsAt540p24(specFixture(), "ltx-2.3-fast"),
      [],
    );
    assert.equal(a2vTrimCapSeconds(specFixture(), "ltx-2.3-fast"), null);
  });

  it("lists advertised A2V resolutions for the selected offering", () => {
    assert.deepEqual(a2vResolutionsForOffering(specFixture(), "ltx-2.5-fast"), [
      "270p",
      "360p",
      "540p",
      "720p",
      "1080p",
    ]);
    assert.deepEqual(
      advertisedA2vDurations(specFixture(), "ltx-2.5-fast", "1080p"),
      [5, 10],
    );
  });

  it("keeps a 20s clip's first 10s at 1080p and the whole clip at 540p", () => {
    const specs = specFixture();
    const longest = a2vLongestCellSeconds(specs, "ltx-2.5-fast");
    assert.equal(longest, 20);
    assert.equal(
      a2vEffectiveAudioSeconds(
        20,
        advertisedA2vDurations(specs, "ltx-2.5-fast", "1080p"),
        longest,
      ),
      10,
    );
    assert.equal(
      a2vEffectiveAudioSeconds(
        20,
        advertisedA2vDurations(specs, "ltx-2.5-fast", "540p"),
        longest,
      ),
      20,
    );
    assert.equal(
      a2vEffectiveAudioSeconds(20.1, [5, 10], longest),
      null,
    );
  });
});
