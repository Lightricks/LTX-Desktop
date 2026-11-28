import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";

import {
  DEFAULT_CUTOUT_COLOR,
  isHexColor,
  pressColorSwatch,
  useCutoutBackgroundStore,
} from "./cutoutBackgroundStore.ts";

describe("cutout background", () => {
  afterEach(() => {
    useCutoutBackgroundStore.setState({
      background: "checkerboard",
      color: DEFAULT_CUTOUT_COLOR,
    });
  });

  it("starts on the transparency grid with a black swatch", () => {
    const state = useCutoutBackgroundStore.getState();
    assert.equal(state.background, "checkerboard");
    assert.equal(state.color, "#000000");
  });

  it("selects an unselected color swatch and opens the picker on the next press", () => {
    assert.deepEqual(pressColorSwatch("checkerboard"), {
      background: "color",
      openPicker: false,
    });
    assert.deepEqual(pressColorSwatch("color"), {
      background: "color",
      openPicker: true,
    });
  });

  it("keeps the picked color when the user goes back to the grid", () => {
    const store = useCutoutBackgroundStore.getState();
    store.setColor("#FF00FF");
    assert.equal(useCutoutBackgroundStore.getState().background, "color");
    assert.equal(useCutoutBackgroundStore.getState().color, "#ff00ff");

    useCutoutBackgroundStore.getState().selectCheckerboard();
    assert.equal(useCutoutBackgroundStore.getState().background, "checkerboard");
    useCutoutBackgroundStore.getState().selectColor();
    assert.equal(useCutoutBackgroundStore.getState().color, "#ff00ff");
  });

  it("ignores a value that is not a six digit hex color", () => {
    for (const value of ["", "red", "#fff", "#12345g", "#1234567"]) {
      assert.equal(isHexColor(value), false, value);
      useCutoutBackgroundStore.getState().setColor(value);
    }
    assert.equal(useCutoutBackgroundStore.getState().background, "checkerboard");
    assert.equal(useCutoutBackgroundStore.getState().color, DEFAULT_CUTOUT_COLOR);
  });
});
