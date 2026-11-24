import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  audioPlaybackProgress,
  shouldPlayAudioPreview,
} from "./audioTilePreview.ts";

describe("shouldPlayAudioPreview", () => {
  it("plays only when hover preview is enabled and the tile is hovered", () => {
    assert.equal(shouldPlayAudioPreview(true, true), true);
  });

  it("keeps idle tiles silent so the waveform can show without playback", () => {
    assert.equal(shouldPlayAudioPreview(false, true), false);
  });

  it("does not play when hover preview is off, even if the tile is hovered", () => {
    assert.equal(shouldPlayAudioPreview(true, false), false);
    assert.equal(shouldPlayAudioPreview(false, false), false);
  });
});

describe("audioPlaybackProgress", () => {
  it("is 0 until duration is a positive finite number", () => {
    assert.equal(audioPlaybackProgress(1, 0), 0);
    assert.equal(audioPlaybackProgress(1, Number.NaN), 0);
    assert.equal(audioPlaybackProgress(1, Number.POSITIVE_INFINITY), 0);
  });

  it("returns currentTime / duration clamped to 0–1", () => {
    assert.equal(audioPlaybackProgress(2.5, 10), 0.25);
    assert.equal(audioPlaybackProgress(-1, 10), 0);
    assert.equal(audioPlaybackProgress(12, 10), 1);
  });
});
