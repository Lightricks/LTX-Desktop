import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { preflightCellStatus } from "../../lib/budget-risk.ts";

import {
  type EditPreflightState,
  preflightFallbackMessage,
  selectFreshPreflight,
  selectTrimMaxSeconds,
} from "./editPreflight.ts";

function preflight(overrides: Partial<EditPreflightState> = {}): EditPreflightState {
  return {
    assetId: "video-1",
    response: {
      source: { width: 1920, height: 1080, fps: 24, frames: 481, durationSeconds: 20 },
      resolutions: [
        { key: "original", width: 1920, height: 1056, maxDurationSeconds: 6 },
        { key: "720", width: 1280, height: 704, maxDurationSeconds: 14.5 },
      ],
      cells: [],
    },
    ...overrides,
  };
}

describe("preflightFallbackMessage", () => {
  it("uses the unmeasured fallback for a warned unmeasured cell with no message", () => {
    const status = preflightCellStatus({ ok: true, tier: "unmeasured", message: null });
    assert.equal(preflightFallbackMessage(status), "Unmeasured on your hardware.");
  });
});

describe("selectFreshPreflight", () => {
  it("drops verdicts requested for another video", () => {
    assert.equal(selectFreshPreflight(preflight(), "video-2"), null);
  });

  it("keeps verdicts for the selected video", () => {
    const state = preflight();
    assert.equal(selectFreshPreflight(state, "video-1"), state);
  });

  it("stays null without a selected video", () => {
    assert.equal(selectFreshPreflight(preflight(), null), null);
  });
});

describe("selectTrimMaxSeconds", () => {
  it("withholds a max below the minimum so the slider stays whole", () => {
    const state = preflight({
      response: {
        source: { width: 1920, height: 1080, fps: 24, frames: 481, durationSeconds: 20 },
        resolutions: [
          { key: "original", width: 1920, height: 1056, maxDurationSeconds: 0 },
        ],
        cells: [],
      },
    });
    assert.equal(selectTrimMaxSeconds(state, "original", 2), null);
  });

  it("clamps to the longest passing duration at the current tier", () => {
    assert.equal(selectTrimMaxSeconds(preflight(), "720", 2), 14.5);
  });

  it("falls back to the video length while unknown", () => {
    assert.equal(selectTrimMaxSeconds(null, "original", 2), null);
    assert.equal(selectTrimMaxSeconds(preflight(), "540", 2), null);
  });
});
