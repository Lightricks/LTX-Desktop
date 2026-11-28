import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  clampPlaybackOffset,
  finitePlaybackDuration,
  playbackProgress,
  recordingStopOutcome,
} from "./hooks.ts";

describe("finitePlaybackDuration", () => {
  it("prefers a decoded duration over the recording counter", () => {
    assert.equal(finitePlaybackDuration(1.9, 1), 1.9);
  });

  it("falls back to recorded elapsed when media duration is missing", () => {
    assert.equal(finitePlaybackDuration(Number.POSITIVE_INFINITY, 3), 3);
    assert.equal(finitePlaybackDuration(Number.NaN, 3), 3);
    assert.equal(finitePlaybackDuration(0, 3), 3);
  });
});

describe("playbackProgress", () => {
  it("does not divide by Infinity or NaN", () => {
    assert.equal(playbackProgress(1.5, Number.POSITIVE_INFINITY, 3), 0.5);
    assert.equal(playbackProgress(1.5, Number.NaN, 3), 0.5);
    assert.equal(playbackProgress(0, Number.POSITIVE_INFINITY, 0), 0);
  });
});

describe("clampPlaybackOffset", () => {
  it("keeps a valid offset inside the buffer", () => {
    assert.equal(clampPlaybackOffset(1.5, 3), 1.5);
    assert.equal(clampPlaybackOffset(5, 3), 3);
    assert.equal(clampPlaybackOffset(-1, 3), 0);
  });

  it("returns 0 when duration is missing", () => {
    assert.equal(clampPlaybackOffset(1, 0), 0);
    assert.equal(clampPlaybackOffset(1, Number.NaN), 0);
  });
});

describe("recordingStopOutcome", () => {
  it("does not deliver a file after abort", () => {
    assert.equal(recordingStopOutcome(true, 2), "abort");
  });

  it("reports empty when stop fires with no chunks", () => {
    assert.equal(recordingStopOutcome(false, 0), "empty");
  });

  it("completes when stop fires with data", () => {
    assert.equal(recordingStopOutcome(false, 1), "complete");
  });
});
