import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  peakClipStates,
  peakVideoPreload,
  nextPeakPlayableIndex,
  resolvePeakActiveIndex,
  shouldAttachPeakVideoSrc,
  shouldRestorePeakPoster,
} from "./peakCinematicPlaylistMedia.ts";

const VISIBLE = { hasBeenVisible: true, inView: true, reducedMotion: false };

describe("peakCinematicPlaylistMedia", () => {
  it("does not attach src before the playlist has been visible", () => {
    assert.equal(
      shouldAttachPeakVideoSrc({
        hasBeenVisible: false,
        inView: false,
        isActive: true,
        hasPlaylist: true,
        index: 0,
        activeIndex: 0,
        length: 4,
      }),
      false,
    );
  });

  it("attaches src for the active clip once visible and in view", () => {
    assert.equal(
      shouldAttachPeakVideoSrc({
        hasBeenVisible: true,
        inView: true,
        isActive: true,
        hasPlaylist: true,
        index: 0,
        activeIndex: 0,
        length: 4,
      }),
      true,
    );
  });

  it("attaches src for active and adjacent clips once visible and in view", () => {
    assert.equal(
      shouldAttachPeakVideoSrc({
        hasBeenVisible: true,
        inView: true,
        isActive: true,
        hasPlaylist: true,
        index: 1,
        activeIndex: 1,
        length: 3,
      }),
      true,
    );
    assert.equal(
      shouldAttachPeakVideoSrc({
        hasBeenVisible: true,
        inView: true,
        isActive: false,
        hasPlaylist: true,
        index: 2,
        activeIndex: 1,
        length: 3,
      }),
      true,
    );
    assert.equal(
      shouldAttachPeakVideoSrc({
        hasBeenVisible: true,
        inView: true,
        isActive: false,
        hasPlaylist: true,
        index: 0,
        activeIndex: 1,
        length: 3,
      }),
      true,
    );
    assert.equal(
      shouldAttachPeakVideoSrc({
        hasBeenVisible: false,
        inView: true,
        isActive: true,
        hasPlaylist: true,
        index: 0,
        activeIndex: 0,
        length: 3,
      }),
      false,
    );
  });

  it("does not attach src after leaving the viewport", () => {
    assert.equal(
      shouldAttachPeakVideoSrc({
        hasBeenVisible: true,
        inView: false,
        isActive: true,
        hasPlaylist: true,
        index: 0,
        activeIndex: 0,
        length: 3,
      }),
      false,
    );
  });

  it("wraps adjacent prefetch from the first clip to the last", () => {
    assert.equal(
      shouldAttachPeakVideoSrc({
        hasBeenVisible: true,
        inView: true,
        isActive: false,
        hasPlaylist: true,
        index: 2,
        activeIndex: 0,
        length: 3,
      }),
      true,
    );
  });

  it("maps preload from active/attach state", () => {
    assert.equal(peakVideoPreload(false, false), "none");
    assert.equal(peakVideoPreload(false, true), "metadata");
    assert.equal(peakVideoPreload(true, true), "auto");
  });

  it("does not preload the active clip while its src is detached", () => {
    assert.equal(peakVideoPreload(true, false), "none");
  });

  it("restores the poster only when a clip loses an attached src", () => {
    assert.equal(shouldRestorePeakPoster(true, false), true);
    assert.equal(shouldRestorePeakPoster(true, true), false);
    assert.equal(shouldRestorePeakPoster(false, false), false);
    assert.equal(shouldRestorePeakPoster(false, true), false);
  });
});

describe("resolvePeakActiveIndex", () => {
  it("keeps an in-range index", () => {
    assert.equal(resolvePeakActiveIndex(0, 4), 0);
    assert.equal(resolvePeakActiveIndex(3, 4), 3);
  });

  it("falls back to the first clip when the index is stale", () => {
    assert.equal(resolvePeakActiveIndex(4, 4), 0);
    assert.equal(resolvePeakActiveIndex(9, 3), 0);
    assert.equal(resolvePeakActiveIndex(-1, 3), 0);
  });

  it("returns the first index for an empty playlist", () => {
    assert.equal(resolvePeakActiveIndex(0, 0), 0);
    assert.equal(resolvePeakActiveIndex(2, 0), 0);
  });
});

describe("nextPeakPlayableIndex", () => {
  const ids = ["a", "b", "c", "d"];

  it("skips failed clips and wraps around the playlist", () => {
    assert.equal(nextPeakPlayableIndex(ids, 0, new Set(["b", "c"])), 3);
  });

  it("reuses the current clip when it is the only playable clip", () => {
    assert.equal(nextPeakPlayableIndex(ids, 2, new Set(["a", "b", "d"])), 2);
  });

  it("returns null when every clip has failed", () => {
    assert.equal(nextPeakPlayableIndex(ids, 2, new Set(ids)), null);
  });
});

describe("peakClipStates", () => {
  const ids = ["a", "b", "c", "d"];
  const noFailures: ReadonlySet<string> = new Set();

  it("marks one clip active and attaches the active plus adjacent clips", () => {
    const states = peakClipStates({
      ids,
      activeIndex: 1,
      failedIds: noFailures,
      ...VISIBLE,
    });

    assert.deepEqual(states, [
      { id: "a", index: 0, isActive: false, attachSrc: true },
      { id: "b", index: 1, isActive: true, attachSrc: true },
      { id: "c", index: 2, isActive: false, attachSrc: true },
      { id: "d", index: 3, isActive: false, attachSrc: false },
    ]);
  });

  it("resolves a stale active index consistently for active and adjacent state", () => {
    const states = peakClipStates({
      ids,
      activeIndex: 9,
      failedIds: noFailures,
      ...VISIBLE,
    });

    assert.deepEqual(
      states.map((state) => state.isActive),
      [true, false, false, false],
    );
    // Adjacency wraps around the resolved index, not the stale one.
    assert.deepEqual(
      states.map((state) => state.attachSrc),
      [true, true, false, true],
    );
  });

  it("never attaches a src while reduced motion is requested", () => {
    const states = peakClipStates({
      ids,
      activeIndex: 0,
      failedIds: noFailures,
      hasBeenVisible: true,
      inView: true,
      reducedMotion: true,
    });

    assert.deepEqual(
      states.map((state) => state.attachSrc),
      [false, false, false, false],
    );
    assert.deepEqual(
      states.map((state) => state.isActive),
      [true, false, false, false],
    );
  });

  it("never attaches a src for a clip that already failed", () => {
    const states = peakClipStates({
      ids,
      activeIndex: 0,
      failedIds: new Set(["a"]),
      ...VISIBLE,
    });

    assert.deepEqual(
      states.map((state) => state.attachSrc),
      [false, true, false, true],
    );
  });

  it("attaches nothing before the playlist has been visible or while out of view", () => {
    for (const visibility of [
      { hasBeenVisible: false, inView: false },
      { hasBeenVisible: false, inView: true },
      { hasBeenVisible: true, inView: false },
    ]) {
      const states = peakClipStates({
        ids,
        activeIndex: 0,
        failedIds: noFailures,
        reducedMotion: false,
        ...visibility,
      });
      assert.deepEqual(
        states.map((state) => state.attachSrc),
        [false, false, false, false],
      );
    }
  });

  it("attaches only the single clip of a one-clip playlist", () => {
    const states = peakClipStates({
      ids: ["solo"],
      activeIndex: 0,
      failedIds: noFailures,
      ...VISIBLE,
    });

    assert.deepEqual(states, [
      { id: "solo", index: 0, isActive: true, attachSrc: true },
    ]);
  });

  it("returns no states for an empty playlist", () => {
    assert.deepEqual(
      peakClipStates({
        ids: [],
        activeIndex: 0,
        failedIds: noFailures,
        ...VISIBLE,
      }),
      [],
    );
  });
});
