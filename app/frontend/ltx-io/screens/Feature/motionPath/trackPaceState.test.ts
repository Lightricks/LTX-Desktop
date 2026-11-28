import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { MotionPath } from "../../../lib/motionPathSampler.ts";
import {
  initialTrackPaceState,
  paceOfTrack,
  withDuration,
  withPaceUpdate,
  withSamePace,
} from "./trackPaceState.ts";

function path(overrides: Partial<MotionPath> = {}): MotionPath {
  return {
    points: [
      { id: "a", x: 0.1, y: 0.1 },
      { id: "b", x: 0.9, y: 0.9 },
    ],
    easing: "linear",
    startHoldSec: 0,
    endHoldSec: 0,
    ...overrides,
  };
}

const IDS = [0, 1];

describe("track pace state", () => {
  it("starts shared when every saved track has the same pace, and individual when not", () => {
    assert.equal(initialTrackPaceState([path(), path()], 4).samePace, true);
    assert.equal(initialTrackPaceState([], 4).samePace, true);
    const mixed = initialTrackPaceState([path(), path({ easing: "slow", endHoldSec: 1 })], 4);
    assert.equal(mixed.samePace, false);
    assert.equal(paceOfTrack(mixed, 0).easing, "linear");
    assert.deepEqual(paceOfTrack(mixed, 1), { easing: "slow", startHoldSec: 0, endHoldSec: 1 });
  });

  it("changes every track at once while the pace is shared", () => {
    const state = withPaceUpdate(initialTrackPaceState([path(), path()], 4), 1, { easing: "quick" });
    assert.equal(paceOfTrack(state, 0).easing, "quick");
    assert.equal(paceOfTrack(state, 1).easing, "quick");
  });

  it("changes one track while the paces are individual", () => {
    const individual = withSamePace(initialTrackPaceState([path(), path()], 4), false, 0, IDS);
    const state = withPaceUpdate(individual, 1, { easing: "quick", startHoldSec: 1 });
    assert.deepEqual(paceOfTrack(state, 0), { easing: "linear", startHoldSec: 0, endHoldSec: 0 });
    assert.deepEqual(paceOfTrack(state, 1), { easing: "quick", startHoldSec: 1, endHoldSec: 0 });
  });

  it("makes the shared pace the pace of the track on screen, and restarts from it after", () => {
    const mixed = initialTrackPaceState([path(), path({ easing: "slow" })], 4);
    const shared = withSamePace(mixed, true, 1, IDS);
    assert.equal(paceOfTrack(shared, 0).easing, "slow");
    assert.equal(paceOfTrack(shared, 1).easing, "slow");
    const edited = withPaceUpdate(shared, 0, { easing: "bouncy" });
    const individual = withSamePace(edited, false, 0, IDS);
    // The old individual paces (linear, slow) must not come back.
    assert.equal(paceOfTrack(individual, 0).easing, "bouncy");
    assert.equal(paceOfTrack(individual, 1).easing, "bouncy");
  });

  it("keeps the holds of every track under the clip length", () => {
    const individual = withSamePace(initialTrackPaceState([path(), path()], 8), false, 0, IDS);
    const held = withPaceUpdate(withPaceUpdate(individual, 1, { startHoldSec: 3 }), 1, {
      endHoldSec: 3,
    });
    const shorter = withDuration(held, 4);
    for (const id of IDS) {
      const { startHoldSec, endHoldSec } = paceOfTrack(shorter, id);
      assert.ok(startHoldSec + endHoldSec < 4, `track ${id}`);
    }
    // Only the track with holds changed.
    assert.deepEqual(paceOfTrack(shorter, 0), { easing: "linear", startHoldSec: 0, endHoldSec: 0 });
  });
});
