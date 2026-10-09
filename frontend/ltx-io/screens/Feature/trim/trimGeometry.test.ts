import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { MIN_TRIM_DURATION_SECONDS } from "./trimConstants.ts";
import {
  clampTrimRange,
  formatTrimTimecode,
  initialTrimRange,
  isApplicableTrimRange,
  pixelsToTime,
  playheadWindow,
  snapSelectionToNeedle,
  timeToPixels,
  trimMediaWidth,
  trimSelectionRect,
} from "./trimGeometry.ts";

describe("trim selection floor", () => {
  it("stays at 6s after generation offers 2s clips", () => {
    assert.equal(MIN_TRIM_DURATION_SECONDS, 6);
  });

  it("keeps a drag from shrinking the selection under the floor", () => {
    assert.deepEqual(
      clampTrimRange(
        { startSec: 0, endSec: 0.5 },
        { durationSeconds: 60, maxDurationSeconds: 20, anchor: "end" },
      ),
      { startSec: 0, endSec: MIN_TRIM_DURATION_SECONDS },
    );
  });
});

describe("initialTrimRange", () => {
  it("opens on the first third when it fits between the floor and the cap", () => {
    assert.deepEqual(initialTrimRange(30, 20), { startSec: 0, endSec: 10 });
  });

  it("opens capped at the max when a third would exceed it", () => {
    assert.deepEqual(initialTrimRange(300, 20), { startSec: 0, endSec: 20 });
  });

  it("opens at the 6s floor when a third would be shorter, for any cap N", () => {
    // A 12s clip under a 20s cap: a third is 4s, which Done would reject.
    assert.deepEqual(initialTrimRange(12, 20), { startSec: 0, endSec: 6 });
    assert.deepEqual(initialTrimRange(12, 8), { startSec: 0, endSec: 6 });
  });

  it("never selects past a clip shorter than the floor", () => {
    assert.deepEqual(initialTrimRange(4, 20), { startSec: 0, endSec: 4 });
  });
});

describe("isApplicableTrimRange", () => {
  const bounds = { durationSeconds: 60, maxDurationSeconds: 20 };

  it("rejects a selection under the 6s floor", () => {
    assert.equal(
      isApplicableTrimRange({ startSec: 0, endSec: 5.9 }, bounds),
      false,
    );
    assert.equal(isApplicableTrimRange({ startSec: 0, endSec: 6 }, bounds), true);
  });

  it("rejects a selection longer than the cap", () => {
    assert.equal(
      isApplicableTrimRange({ startSec: 0, endSec: 20.5 }, bounds),
      false,
    );
  });

  it("rejects a selection that runs past the clip", () => {
    assert.equal(
      isApplicableTrimRange({ startSec: 50, endSec: 65 }, bounds),
      false,
    );
  });

  it("allows the whole clip when it is shorter than the floor", () => {
    assert.equal(
      isApplicableTrimRange(
        { startSec: 0, endSec: 4 },
        { durationSeconds: 4, maxDurationSeconds: 20 },
      ),
      true,
    );
  });
});

describe("clampTrimRange", () => {
  const bounds = { durationSeconds: 60, maxDurationSeconds: 20 };

  it("holds the Out edge while dragging In past the max window", () => {
    assert.deepEqual(
      clampTrimRange({ startSec: 0, endSec: 40 }, { ...bounds, anchor: "start" }),
      { startSec: 20, endSec: 40 },
    );
  });

  it("holds the In edge while dragging Out past the max window", () => {
    assert.deepEqual(
      clampTrimRange({ startSec: 10, endSec: 55 }, { ...bounds, anchor: "end" }),
      { startSec: 10, endSec: 30 },
    );
  });

  it("keeps the minimum window when the handles cross", () => {
    const range = clampTrimRange(
      { startSec: 30, endSec: 10 },
      { ...bounds, anchor: "end", minDurationSeconds: 1 },
    );

    assert.deepEqual(range, { startSec: 30, endSec: 31 });
  });

  it("slides a whole selection back inside the clip", () => {
    assert.deepEqual(
      clampTrimRange({ startSec: 55, endSec: 70 }, { ...bounds, anchor: "both" }),
      { startSec: 45, endSec: 60 },
    );
  });

  it("never selects past the clip when the cap is longer than the clip", () => {
    const range = clampTrimRange(
      { startSec: 0, endSec: 30 },
      { durationSeconds: 8, maxDurationSeconds: 20, anchor: "end" },
    );

    assert.deepEqual(range, { startSec: 0, endSec: 8 });
  });
});

describe("snapSelectionToNeedle", () => {
  const bounds = {
    durationSeconds: 30,
    maxDurationSeconds: 10,
    minDurationSeconds: 2,
  };
  const selection = { startSec: 0, endSec: 10 };

  it("starts the selection at the needle and keeps its duration", () => {
    assert.deepEqual(snapSelectionToNeedle(8, selection, bounds), {
      startSec: 8,
      endSec: 18,
    });
    assert.deepEqual(
      snapSelectionToNeedle(4, { startSec: 0, endSec: 6 }, bounds),
      { startSec: 4, endSec: 10 },
    );
  });

  it("does not snap when the tail is shorter than the minimum", () => {
    assert.equal(snapSelectionToNeedle(29, selection, bounds), null);
    assert.equal(snapSelectionToNeedle(28.1, selection, bounds), null);
  });

  it("shortens the selection to the tail when that tail still meets the minimum", () => {
    assert.deepEqual(snapSelectionToNeedle(26, selection, bounds), {
      startSec: 26,
      endSec: 30,
    });
  });

  it("snaps when the tail is exactly the minimum", () => {
    assert.deepEqual(snapSelectionToNeedle(28, selection, bounds), {
      startSec: 28,
      endSec: 30,
    });
  });

  it("does not snap a needle that sits on the end of the clip", () => {
    assert.equal(snapSelectionToNeedle(30, selection, bounds), null);
    assert.equal(snapSelectionToNeedle(40, selection, bounds), null);
  });

  it("uses the 6s floor when the caller does not pass a minimum", () => {
    const audioBounds = { durationSeconds: 30, maxDurationSeconds: 20 };
    assert.equal(snapSelectionToNeedle(25, selection, audioBounds), null);
    assert.deepEqual(snapSelectionToNeedle(24, selection, audioBounds), {
      startSec: 24,
      endSec: 30,
    });
  });
});

describe("pixel/time conversion", () => {
  it("round-trips a time through pixels", () => {
    assert.equal(pixelsToTime(timeToPixels(3.5, 40), 40), 3.5);
  });
});

describe("formatTrimTimecode", () => {
  it("renders M:SS", () => {
    assert.equal(formatTrimTimecode(0), "0:00");
    assert.equal(formatTrimTimecode(9.9), "0:09");
    assert.equal(formatTrimTimecode(75), "1:15");
  });

  it("treats non-finite input as zero", () => {
    assert.equal(formatTrimTimecode(Number.NaN), "0:00");
    assert.equal(formatTrimTimecode(-5), "0:00");
  });
});

describe("playheadWindow", () => {
  const selection = { startSec: 5, endSec: 8 };

  it("pins A2V trim playback to the In/Out window", () => {
    assert.deepEqual(playheadWindow(undefined, selection), selection);
  });

  it("lets Retake scrub and play the full clip", () => {
    assert.deepEqual(playheadWindow({ startSec: 0, endSec: 12 }, selection), {
      startSec: 0,
      endSec: 12,
    });
  });
});

describe("trim media fill", () => {
  it("lets the filmstrip occupy the full bar, including handle gutters", () => {
    assert.equal(trimMediaWidth(640), 640);
  });

  it("hangs In/Out handles over the frames instead of leaving empty gutters", () => {
    assert.deepEqual(trimSelectionRect(0, 200, 12, 640), {
      left: 0,
      width: 212,
    });
    assert.deepEqual(trimSelectionRect(100, 300, 12, 640), {
      left: 88,
      width: 224,
    });
    assert.deepEqual(trimSelectionRect(620, 640, 12, 640), {
      left: 608,
      width: 32,
    });
  });
});
