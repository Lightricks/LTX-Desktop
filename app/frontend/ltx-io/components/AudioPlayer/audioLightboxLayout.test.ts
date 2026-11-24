import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  AUDIO_LIGHTBOX_CONTROLS_HEIGHT,
  AUDIO_LIGHTBOX_MIN_WAVEFORM_HEIGHT,
  audioLightboxWaveformHeight,
} from "./audioLightboxLayout.ts";

describe("audioLightboxWaveformHeight", () => {
  it("matches ltx.io: max(200, 50% of the area minus the 56px control bar)", () => {
    const areaHeight = 630;
    const expected = Math.max(
      AUDIO_LIGHTBOX_MIN_WAVEFORM_HEIGHT,
      Math.floor((areaHeight - AUDIO_LIGHTBOX_CONTROLS_HEIGHT) * 0.5),
    );
    assert.equal(audioLightboxWaveformHeight(areaHeight), expected);
    assert.equal(audioLightboxWaveformHeight(areaHeight), 287);
  });

  it("never goes below 200px even in a short stage", () => {
    assert.equal(audioLightboxWaveformHeight(100), AUDIO_LIGHTBOX_MIN_WAVEFORM_HEIGHT);
  });
});
