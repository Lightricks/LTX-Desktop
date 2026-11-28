import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { MOTION_PATH_MAX_POINTS } from "../../../lib/motionPathSampler.ts";
import { commitFreehandStroke } from "./freehandToMotionPath.ts";
import {
  beginMotionPathPointerDown,
  stepMotionPathPointerMove,
} from "./motionPathGestures.ts";

describe("free draw", () => {
  it("keeps a long wiggly stroke to the most points the backend accepts", () => {
    const stroke = Array.from({ length: 300 }, (_, index) => ({
      x: index / 299,
      y: 0.5 + 0.4 * Math.sin(index / 8),
    }));

    const points = commitFreehandStroke(stroke);

    assert.ok(points != null);
    assert.ok(points.length >= 2);
    assert.ok(points.length <= MOTION_PATH_MAX_POINTS);
  });
});

describe("motion path gestures", () => {
  it("curves a newly placed point only after dragging past the threshold and clears handles if dragged back", () => {
    const initial = [{ id: "p0", x: 0.2, y: 0.2 }];
    const down = beginMotionPathPointerDown({
      points: initial,
      tool: "pen",
      selectedPointId: "p0",
      local: { px: 100, py: 100, width: 200, height: 200 },
    });
    assert.equal(down.gesture.kind, "place");
    const placed = down.pointsEdit?.points ?? [];
    assert.equal(placed.length, 2);

    const movedFar = stepMotionPathPointerMove(down.gesture, placed, {
      px: 120,
      py: 100,
      width: 200,
      height: 200,
    });
    const curved = movedFar.pointsEdit?.points ?? [];
    assert.ok(curved[1]?.handleIn != null && curved[1]?.handleOut != null);

    const movedBack = stepMotionPathPointerMove(movedFar.gesture, curved, {
      px: 102,
      py: 100,
      width: 200,
      height: 200,
    });
    const straightened = movedBack.pointsEdit?.points ?? [];
    assert.equal(straightened[1]?.handleIn, undefined);
    assert.equal(straightened[1]?.handleOut, undefined);
  });

  it("pushes history on the first drag move of a point and replaces on later moves", () => {
    const points = [
      { id: "p0", x: 0.2, y: 0.2 },
      { id: "p1", x: 0.8, y: 0.8 },
    ];
    const down = beginMotionPathPointerDown({
      points,
      tool: "pen",
      selectedPointId: null,
      local: { px: 40, py: 40, width: 200, height: 200 },
    });
    assert.equal(down.gesture.kind, "dragPoint");

    const firstMove = stepMotionPathPointerMove(down.gesture, points, {
      px: 60,
      py: 60,
      width: 200,
      height: 200,
    });
    assert.equal(firstMove.pointsEdit?.historyMode, "push");

    const secondMove = stepMotionPathPointerMove(
      firstMove.gesture,
      firstMove.pointsEdit?.points ?? points,
      { px: 80, py: 80, width: 200, height: 200 },
    );
    assert.equal(secondMove.pointsEdit?.historyMode, "replace");
  });
});
