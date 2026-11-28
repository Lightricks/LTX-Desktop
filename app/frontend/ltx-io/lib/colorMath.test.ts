import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { hexToHsv, hsvToHex } from "./colorMath.ts";

describe("color math", () => {
  it("maps the primaries and the grays", () => {
    assert.deepEqual(hexToHsv("#ff0000"), { h: 0, s: 1, v: 1 });
    assert.deepEqual(hexToHsv("#00ff00"), { h: 120, s: 1, v: 1 });
    assert.deepEqual(hexToHsv("#0000ff"), { h: 240, s: 1, v: 1 });
    assert.deepEqual(hexToHsv("#000000"), { h: 0, s: 0, v: 0 });
    assert.deepEqual(hexToHsv("#ffffff"), { h: 0, s: 0, v: 1 });
  });

  it("round trips every color on a coarse grid", () => {
    for (const red of [0, 51, 128, 255]) {
      for (const green of [0, 77, 200, 255]) {
        for (const blue of [0, 9, 130, 255]) {
          const hex = `#${[red, green, blue].map((c) => c.toString(16).padStart(2, "0")).join("")}`;
          assert.equal(hsvToHex(hexToHsv(hex)), hex);
        }
      }
    }
  });

  it("wraps the hue and clamps saturation and value", () => {
    assert.equal(hsvToHex({ h: 360, s: 1, v: 1 }), "#ff0000");
    assert.equal(hsvToHex({ h: -120, s: 1, v: 1 }), "#0000ff");
    assert.equal(hsvToHex({ h: 0, s: 2, v: 2 }), "#ff0000");
    assert.equal(hsvToHex({ h: 0, s: -1, v: 0.5 }), "#808080");
  });
});
