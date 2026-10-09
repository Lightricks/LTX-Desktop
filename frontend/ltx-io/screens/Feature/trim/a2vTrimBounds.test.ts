import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  A2V_TRIM_MIN_SECONDS,
  a2vTrimCapSeconds,
} from "../../../lib/a2vDurationPolicy.ts";
import type {
  VideoGenerationDuration,
  VideoGenerationModelSpecsResponse,
} from "../../../../lib/video-generation-model-specs.ts";
import { MIN_TRIM_DURATION_SECONDS } from "./trimConstants.ts";
import {
  clampTrimRange,
  initialTrimRange,
  isApplicableTrimRange,
} from "./trimGeometry.ts";

function a2vSpecs(
  durations: VideoGenerationDuration[],
): VideoGenerationModelSpecsResponse {
  const spec = {
    display_name: "LTX 2.5 Fast",
    supported_resolutions_durations: {},
    a2v_supported_resolutions_durations: {
      "540p": {
        fps_to_durations: { "24": durations },
        aspect_ratios: ["21:9", "16:9", "3:2", "4:3", "1:1", "4:5", "9:16"],
      },
    },
  };
  return {
    api_models: [],
    local_models: [{ pipeline: "fast", spec }],
    downloaded_local_models: [
      { model: "ltx-2.5-fast", pipeline: "fast", spec },
    ],
  };
}

describe("A2V trim bounds", () => {
  it("uses the 2s audio floor, not the generic 6s selection floor", () => {
    assert.equal(A2V_TRIM_MIN_SECONDS, 2);
    // Generic GenSpace behavior stays put: non-A2V trims keep the 6s floor.
    assert.equal(MIN_TRIM_DURATION_SECONDS, 6);
  });

  it("keeps a 5s-only envelope usable: selection clamped to 5s, Done enabled", () => {
    const cap = a2vTrimCapSeconds(a2vSpecs([5]));
    assert.equal(cap, 5);
    const bounds = {
      durationSeconds: 31,
      maxDurationSeconds: cap,
      minDurationSeconds: A2V_TRIM_MIN_SECONDS,
    };
    // Opens on the 5s cap instead of a range Done would reject.
    assert.deepEqual(initialTrimRange(31, cap, A2V_TRIM_MIN_SECONDS), {
      startSec: 0,
      endSec: 5,
    });
    // Done's own gate accepts the clamped 5s selection.
    assert.equal(
      isApplicableTrimRange({ startSec: 0, endSec: 5 }, bounds),
      true,
    );
  });

  it("accepts 2-5s selections and rejects under 2s", () => {
    const bounds = {
      durationSeconds: 31,
      maxDurationSeconds: 5,
      minDurationSeconds: A2V_TRIM_MIN_SECONDS,
    };
    assert.equal(
      isApplicableTrimRange({ startSec: 0, endSec: 2 }, bounds),
      true,
    );
    assert.equal(
      isApplicableTrimRange({ startSec: 1, endSec: 5 }, bounds),
      true,
    );
    assert.equal(
      isApplicableTrimRange({ startSec: 0, endSec: 1.9 }, bounds),
      false,
    );
  });

  it("holds the 2s window when the handles shrink past it", () => {
    assert.deepEqual(
      clampTrimRange(
        { startSec: 0, endSec: 0.5 },
        {
          durationSeconds: 31,
          maxDurationSeconds: 5,
          minDurationSeconds: A2V_TRIM_MIN_SECONDS,
          anchor: "end",
        },
      ),
      { startSec: 0, endSec: 2 },
    );
  });

  it("documents why the generic floor breaks A2V: 5s rejected under the 6s default", () => {
    // What the A2V modal did before it passed its own 2s min: every 5s
    // selection failed Done's gate, so a 5s-only envelope had no usable range.
    assert.equal(
      isApplicableTrimRange(
        { startSec: 0, endSec: 5 },
        { durationSeconds: 31, maxDurationSeconds: 5 },
      ),
      false,
    );
  });
});
