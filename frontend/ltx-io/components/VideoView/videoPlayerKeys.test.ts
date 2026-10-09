import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  seekByNativeVideoStep,
  shouldIgnorePlayerToggle,
  videoPlayerKeyAction,
} from "./videoPlayerKeys.ts";

describe("videoPlayerKeyAction", () => {
  it("maps space to play and pause, and arrows to a native seek", () => {
    assert.equal(videoPlayerKeyAction(" "), "toggle-playback");
    assert.equal(videoPlayerKeyAction("ArrowLeft"), "seek-backward");
    assert.equal(videoPlayerKeyAction("ArrowRight"), "seek-forward");
    assert.equal(videoPlayerKeyAction("ArrowUp"), null);
  });
});

describe("shouldIgnorePlayerToggle", () => {
  it("lets a focused control button keep Space, and still seeks from that button", () => {
    assert.equal(shouldIgnorePlayerToggle("toggle-playback", true), true);
    assert.equal(shouldIgnorePlayerToggle("toggle-playback", false), false);
    assert.equal(shouldIgnorePlayerToggle("seek-forward", true), false);
  });
});

describe("seekByNativeVideoStep", () => {
  it("steps one second and stays inside the clip", () => {
    assert.equal(seekByNativeVideoStep(4, 10, 1), 5);
    assert.equal(seekByNativeVideoStep(4, 10, -1), 3);
    assert.equal(seekByNativeVideoStep(0.2, 10, -1), 0);
    assert.equal(seekByNativeVideoStep(9.5, 10, 1), 10);
  });
});
