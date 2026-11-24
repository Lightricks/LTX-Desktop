import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type {
  VideoGenerationDuration,
  VideoGenerationModelSpecsResponse,
} from "../../lib/video-generation-model-specs.ts";
import { specFixture } from "../screens/Feature/definitions/specFixture.ts";

import {
  a2vDisabledResolutionsForOffering,
  a2vDurationForFrames,
  a2vEffectiveAudioSeconds,
  a2vResolutionsForOffering,
  a2vTrimCapSeconds,
  advertisedA2vDurations,
  advertisedA2vDurationsAt540p24,
  computeA2vMaxFrames,
  formatA2vDurationBadge,
  hasAdvertisedA2vCell,
  numFramesForAudioDuration,
  resolveA2vAudio,
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
    assert.deepEqual(
      advertisedA2vDurationsAt540p24(specs, "ltx-2.5-fast"),
      [5, 6, 8, 10, 20],
    );
    assert.equal(a2vTrimCapSeconds(specs, "ltx-2.5-fast"), 20);
    assert.deepEqual(advertisedA2vDurationsAt540p24(specs, "ltx-2.3-fast"), [5]);
    assert.equal(a2vTrimCapSeconds(specs, "ltx-2.3-fast"), 5);
    // No offering: Settings-matching downloaded row (2.3).
    assert.deepEqual(advertisedA2vDurationsAt540p24(specs), [5]);
    assert.equal(a2vTrimCapSeconds(specs), 5);
  });

  it("reads advertised A2V durations at 540p/24 from local fast specs", () => {
    assert.deepEqual(advertisedA2vDurationsAt540p24(specFixture()), [5, 6, 8, 10, 20]);
    assert.deepEqual(
      advertisedA2vDurationsAt540p24(a2vSpecs([5, 6, 8, 10, 20])),
      [5, 6, 8, 10, 20],
    );
  });

  it("trim cap is the max advertised duration", () => {
    assert.equal(a2vTrimCapSeconds(specFixture()), 20);
    assert.equal(a2vTrimCapSeconds(null), null);
    assert.equal(a2vTrimCapSeconds(a2vSpecs([5, 6, 8, 10, 20])), 20);
  });

  it("lists dropped-only A2V resolutions with the smallest-duration message", () => {
    const specs = a2vSpecs([5]);
    specs.downloaded_local_models[0].spec.dropped_cells = [
      {
        resolution: "720p",
        fps: 24,
        duration: 5,
        reason: "stream_over",
        message: "m720",
        audio: true,
      },
      {
        resolution: "720p",
        fps: 24,
        duration: 10,
        reason: "stream_over",
        message: "m720b",
        audio: true,
      },
    ];
    assert.deepEqual(a2vDisabledResolutionsForOffering(specs, "ltx-2.5-fast"), [
      { value: "720p", message: "m720" },
    ]);
  });

  it("excludes kept resolutions and plain-matrix drops", () => {
    const specs = a2vSpecs([5]);
    specs.downloaded_local_models[0].spec.dropped_cells = [
      {
        resolution: "540p",
        fps: 24,
        duration: 20,
        reason: "stream_over",
        message: "kept-anyway",
        audio: true,
      },
      {
        resolution: "1080p",
        fps: 24,
        duration: 5,
        reason: "stream_over",
        message: "plain",
        audio: false,
      },
    ];
    assert.deepEqual(a2vDisabledResolutionsForOffering(specs, "ltx-2.5-fast"), []);
  });

  it("floors 10.563s onto 249 frames under the 20s envelope, not 481", () => {
    assert.equal(numFramesForAudioDuration(10.563, 24, 481), 249);
  });

  it("reports over-cap when floored frames exceed the envelope", () => {
    assert.equal(numFramesForAudioDuration(10.563, 24, 241), null);
  });

  it("prices 241 frames as the 10s cell when 20s is also offered", () => {
    assert.equal(a2vDurationForFrames([5, 10, 20], 241), 10);
    assert.equal(a2vDurationForFrames([5, 10, 20], 249), 20);
    assert.equal(a2vDurationForFrames([5, 10], null), null);
  });

  it("treats 10.2s as the kept 10s cell when a longer static duration exists", () => {
    const fit = resolveA2vAudio({
      audioSeconds: 10.2,
      keptSeconds: [5, 10],
      ceilingSeconds: [5, 10, 20],
      longestCeilingSeconds: 20,
    });
    assert.equal(fit.pastKept, false);
    assert.equal(fit.frames, 241);
    assert.equal(fit.heardSeconds, 10.2);
  });

  it("marks 10.4s past the kept 10s cell because its frames no longer fit", () => {
    const fit = resolveA2vAudio({
      audioSeconds: 10.4,
      keptSeconds: [5, 10],
      ceilingSeconds: [5, 10, 20],
      longestCeilingSeconds: 20,
    });
    assert.equal(fit.pastKept, true);
    assert.equal(fit.frames, 249);
  });

  it("floors exact 5s to 113, not the T2V 121 cell", () => {
    assert.equal(numFramesForAudioDuration(5.0, 24, 481), 113);
  });

  it("frame grid matches backend frame_math exactly", () => {
    // Cross-language parity: the 8k+1 grid is mirrored in Python
    // (frame_math.num_frames_for_audio_duration / compute_num_frames) with no
    // shared code, so these vectors pin the agreement. Expected values from:
    //   cd backend && uv run python -c "
    //   from frame_math import compute_num_frames, num_frames_for_audio_duration
    //   for d, m in [(10.563,481),(5.0,481),(5.0,113),(5.0,112),(5.04,481),(5.375,481),(2.0,481),(0.05,481),(30.0,481)]: print(d, m, num_frames_for_audio_duration(d, 24, max_frames=m))
    //   for s in [20,5,0]: print(s, compute_num_frames(s,24))"
    const audioVectors: Array<
      [duration: number, maxFrames: number, frames: number | null]
    > = [
      [10.563, 481, 249],
      [5.0, 481, 113],
      [5.0, 113, 113],
      [5.0, 112, null],
      [5.04, 481, 113],
      [5.375, 481, 129],
      [2.0, 481, 41],
      [0.05, 481, 9],
      [30.0, 481, null],
    ];
    for (const [duration, maxFrames, frames] of audioVectors) {
      assert.equal(numFramesForAudioDuration(duration, 24, maxFrames), frames);
    }
    assert.equal(computeA2vMaxFrames(20, 24), 481);
    assert.equal(computeA2vMaxFrames(5, 24), 121);
    assert.equal(computeA2vMaxFrames(0, 24), 9);
  });

  it("keeps the 2s minimum on the 8k+1 grid", () => {
    const frames = numFramesForAudioDuration(2.0, 24, 481);
    assert.ok(frames !== null);
    assert.ok(frames >= 9);
    assert.equal((frames - 1) % 8, 0);
  });

  it("rejects audio under 2s and past the static ceiling", () => {
    const cells = [6, 8, 10, 20];
    assert.equal(
      resolveA2vAudio({
        audioSeconds: 1,
        keptSeconds: cells,
        ceilingSeconds: cells,
        longestCeilingSeconds: 20,
      }).frames,
      null,
    );
    assert.equal(
      resolveA2vAudio({
        audioSeconds: 20.1,
        keptSeconds: cells,
        ceilingSeconds: cells,
        longestCeilingSeconds: 20,
      }).frames,
      null,
    );
    assert.equal(
      resolveA2vAudio({
        audioSeconds: 20.05,
        keptSeconds: cells,
        ceilingSeconds: cells,
        longestCeilingSeconds: 20,
      }).frames,
      481,
    );
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
    const fit = (audioSeconds: number) =>
      resolveA2vAudio({
        audioSeconds,
        keptSeconds: [5],
        ceilingSeconds: [5],
        longestCeilingSeconds: 5,
      });
    assert.equal(fit(2).frames, 41);
    assert.equal(fit(4.5).frames, 105);
    assert.equal(fit(5.05).frames, 121);
    assert.equal(fit(1.9).frames, null);
    assert.equal(fit(5.1).frames, null);
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
    assert.deepEqual(advertisedA2vDurationsAt540p24(specFixture(), "ltx-2.3-fast"), []);
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
    assert.equal(advertisedA2vDurations(specs, "ltx-2.5-fast", "540p").at(-1), 20);
    const longest = 20;
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
    assert.equal(a2vEffectiveAudioSeconds(20.1, [5, 10], longest), null);
  });
});
