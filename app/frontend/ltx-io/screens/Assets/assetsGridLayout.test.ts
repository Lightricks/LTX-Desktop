import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { computeGridScrollMargin } from "./assetsGridLayout.ts";

describe("computeGridScrollMargin", () => {
  it("is the grid's document offset from the scroll root", () => {
    assert.equal(
      computeGridScrollMargin({ top: 200 }, { top: 80 }, 40),
      160,
    );
  });

  it("moves with the grid when the toolbar reflow increases vertical offset", () => {
    const scrollRect = { top: 80 };
    const scrollTop = 40;
    const before = computeGridScrollMargin({ top: 200 }, scrollRect, scrollTop);
    const afterToolbarWrap = computeGridScrollMargin(
      { top: 248 },
      scrollRect,
      scrollTop,
    );

    assert.equal(afterToolbarWrap - before, 48);
  });
});
