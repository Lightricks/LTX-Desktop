import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { placeBesideAnchor } from "./placeBesideAnchor.ts";

const POPOVER = { width: 200, height: 220 };
const VIEWPORT = { width: 1200, height: 800 };

describe("placeBesideAnchor", () => {
  it("opens on the left of the anchor, top aligned", () => {
    const anchor = { left: 600, top: 100, right: 670, bottom: 132 };
    assert.deepEqual(placeBesideAnchor(anchor, POPOVER, VIEWPORT), { left: 392, top: 100 });
  });

  it("opens on the right when the left has no room", () => {
    const anchor = { left: 8, top: 8, right: 70, bottom: 40 };
    assert.deepEqual(placeBesideAnchor(anchor, POPOVER, VIEWPORT), { left: 78, top: 8 });
  });

  it("stays inside the viewport at the bottom edge", () => {
    const anchor = { left: 600, top: 700, right: 670, bottom: 732 };
    assert.deepEqual(placeBesideAnchor(anchor, POPOVER, VIEWPORT), { left: 392, top: 572 });
  });
});
