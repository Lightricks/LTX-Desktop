import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  PLAYHEAD_HANDLE_WIDTH_PX,
  RESIZE_HANDLE_WIDTH,
} from "./trimConstants.ts";
import { trimSelectionRect } from "./trimGeometry.ts";
import {
  createPlayheadLoop,
  placePlayhead,
  playheadPointerEvents,
  snapPlayheadPx,
  type PlayheadFrames,
} from "./trimPlayhead.ts";

describe("snapPlayheadPx", () => {
  it("leaves a playhead far from In/Out where the pointer is", () => {
    assert.equal(snapPlayheadPx(200, 80, 400), 200);
    assert.equal(snapPlayheadPx(20, 80, 400), 20);
  });

  it("sticks to the closer edge within the magnet", () => {
    assert.equal(snapPlayheadPx(88, 80, 400), 80);
    assert.equal(snapPlayheadPx(72, 80, 400), 80);
    assert.equal(snapPlayheadPx(405, 80, 400), 400);
  });

  it("keeps the middle of a narrow selection free", () => {
    assert.equal(snapPlayheadPx(104, 100, 112), 100);
    assert.equal(snapPlayheadPx(105, 100, 110), 105);
  });
});

/** Minimal stand-in for `<audio>`: settable clock plus manual event dispatch. */
function fakeMedia() {
  const listeners = new Map<string, Set<() => void>>();
  return {
    currentTime: 0,
    paused: true,
    addEventListener(type: string, listener: () => void) {
      const set = listeners.get(type) ?? new Set();
      set.add(listener);
      listeners.set(type, set);
    },
    removeEventListener(type: string, listener: () => void) {
      listeners.get(type)?.delete(listener);
    },
    dispatch(type: string) {
      for (const listener of [...(listeners.get(type) ?? [])]) listener();
    },
    listenerCount() {
      return [...listeners.values()].reduce((sum, set) => sum + set.size, 0);
    },
  };
}

/** rAF replacement with a single pending frame the test advances by hand. */
function manualFrames() {
  let next = 1;
  const pending = new Map<number, () => void>();
  const frames: PlayheadFrames = {
    request: (callback) => {
      const handle = next++;
      pending.set(handle, callback);
      return handle;
    },
    cancel: (handle) => {
      pending.delete(handle);
    },
  };
  return {
    frames,
    isRunning: () => pending.size > 0,
    advance(count = 1) {
      for (let i = 0; i < count; i++) {
        const entry = [...pending.entries()][0];
        if (!entry) return;
        pending.delete(entry[0]);
        entry[1]();
      }
    },
  };
}

function loopUnderTest() {
  const media = fakeMedia();
  const clock = manualFrames();
  const painted: number[] = [];
  const loop = createPlayheadLoop({
    media,
    paint: (timeSec) => painted.push(timeSec),
    frames: clock.frames,
  });
  return { media, clock, painted, loop };
}

describe("createPlayheadLoop", () => {
  it("keeps painting after a scrub that ends while playback continues", () => {
    const { media, clock, painted, loop } = loopUnderTest();
    loop.attach();

    media.paused = false;
    media.dispatch("play");
    media.currentTime = 1;
    clock.advance();
    assert.deepEqual(painted, [0, 1]);

    // Drag: the pointer owns the handle, and each seek fires `seeked` while the
    // media is still playing.
    loop.beginScrub();
    media.currentTime = 4;
    media.dispatch("seeked");
    clock.advance();
    assert.deepEqual(painted, [0, 1], "scrubbing must not fight the pointer");

    loop.endScrub();
    media.currentTime = 5;
    clock.advance();

    assert.equal(clock.isRunning(), true, "playhead froze after pointerup");
    assert.deepEqual(painted, [0, 1, 5]);
  });

  it("does not treat a seek during playback as a pause", () => {
    const { media, clock, painted, loop } = loopUnderTest();
    loop.attach();

    media.paused = false;
    media.dispatch("play");
    media.currentTime = 2;
    media.dispatch("seeked");
    clock.advance();

    assert.equal(clock.isRunning(), true);
    assert.deepEqual(painted, [0, 2, 2]);
  });

  it("stops the loop and settles on the exact frame when playback pauses", () => {
    const { media, clock, painted, loop } = loopUnderTest();
    loop.attach();

    media.paused = false;
    media.dispatch("play");
    assert.equal(clock.isRunning(), true);

    media.paused = true;
    media.currentTime = 7;
    media.dispatch("pause");

    assert.equal(clock.isRunning(), false);
    assert.deepEqual(painted, [0, 7]);
  });

  it("settles without restarting the loop when a scrub ends while paused", () => {
    const { media, clock, painted, loop } = loopUnderTest();
    loop.attach();

    loop.beginScrub();
    media.currentTime = 3;
    media.dispatch("seeked");
    loop.endScrub();

    assert.equal(clock.isRunning(), false);
    assert.deepEqual(painted, [0, 3]);
  });

  it("starts already-playing media on attach and cleans up on detach", () => {
    const { media, clock, loop } = loopUnderTest();
    media.paused = false;

    const detach = loop.attach();
    assert.equal(clock.isRunning(), true);
    assert.equal(media.listenerCount(), 3);

    detach();
    assert.equal(clock.isRunning(), false);
    assert.equal(media.listenerCount(), 0);
  });
});

describe("playheadPointerEvents", () => {
  const W = RESIZE_HANDLE_WIDTH;
  const half = PLAYHEAD_HANDLE_WIDTH_PX / 2;
  // In at 100px, Out at 300px: handles span [88, 100] and [300, 312].
  const mid = trimSelectionRect(100, 300, W, 1000);
  // In at the clip's left edge: the rect clamps, so In spans [0, 12].
  const atEdge = trimSelectionRect(0, 200, W, 1000);

  it("lets In win when the playhead parks on In", () => {
    assert.equal(playheadPointerEvents(100, mid), "none");
  });

  it("lets Out win when the playhead parks on Out", () => {
    assert.equal(playheadPointerEvents(300, mid), "none");
  });

  it("captures once the stem clears In or Out", () => {
    assert.equal(playheadPointerEvents(100 + half, mid), "auto");
    assert.equal(playheadPointerEvents(300 - half, mid), "auto");
    assert.equal(playheadPointerEvents(88 - half, mid), "auto");
  });

  it("reads a clamped In handle from the rect, not left of the In time", () => {
    assert.equal(playheadPointerEvents(0, atEdge), "none");
    assert.equal(playheadPointerEvents(half, atEdge), "none");
    assert.equal(playheadPointerEvents(W + half, atEdge), "auto");
  });

  it("reads a clamped Out handle at the clip's right edge", () => {
    const outAtEdge = trimSelectionRect(800, 1000, W, 1000);
    assert.equal(playheadPointerEvents(1000 - half, outAtEdge), "none");
    assert.equal(playheadPointerEvents(1000 - W - half, outAtEdge), "auto");
  });

  it("writes the left edge onto the handle and pointer-events onto the stem", () => {
    const handle = { style: { left: "" } };
    const stem = { style: { pointerEvents: "auto" } };
    placePlayhead(handle, stem, 100, mid);
    assert.equal(handle.style.left, "100px");
    assert.equal(stem.style.pointerEvents, "none");
    placePlayhead(handle, stem, 200, mid);
    assert.equal(stem.style.pointerEvents, "auto");
  });
});
