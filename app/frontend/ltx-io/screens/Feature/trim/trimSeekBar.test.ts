import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

// The playhead position is written via `style.left` outside React and has no
// keyboard controls, so it cannot satisfy the ARIA slider contract. Assert at
// the source level that it stays a pointer-only handle with a label.
const SEEK_BAR_SOURCE = readFileSync(
  fileURLToPath(new URL("./TrimSeekBar.tsx", import.meta.url)),
  "utf8",
);

describe("TrimSeekBar playhead semantics", () => {
  it("does not expose a misleading slider contract", () => {
    assert.doesNotMatch(SEEK_BAR_SOURCE, /role="slider"/);
    assert.doesNotMatch(SEEK_BAR_SOURCE, /aria-valuemin/);
    assert.doesNotMatch(SEEK_BAR_SOURCE, /aria-valuemax/);
    assert.doesNotMatch(SEEK_BAR_SOURCE, /tabIndex/);
  });

  it("keeps the playhead label and pointer scrub behavior", () => {
    assert.match(SEEK_BAR_SOURCE, /aria-label="Playhead"/);
    assert.match(SEEK_BAR_SOURCE, /onPointerDown/);
    assert.match(SEEK_BAR_SOURCE, /onPointerMove/);
    assert.match(SEEK_BAR_SOURCE, /onPointerUp/);
  });
});
