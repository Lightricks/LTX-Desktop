import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";

import {
  activateHoverVideo,
  deactivateHoverVideo,
  resetHoverVideoCoordinator,
} from "./hoverVideoCoordinator.ts";

afterEach(() => {
  resetHoverVideoCoordinator();
});

describe("hoverVideoCoordinator", () => {
  it("stops the previous asset when a different asset becomes active", () => {
    let firstStops = 0;
    let secondStops = 0;

    activateHoverVideo("first", () => {
      firstStops += 1;
    });
    activateHoverVideo("second", () => {
      secondStops += 1;
    });

    assert.equal(firstStops, 1);
    assert.equal(secondStops, 0);
  });

  it("keeps the active video when another tile leaves", () => {
    let activeStops = 0;

    activateHoverVideo("active", () => {
      activeStops += 1;
    });
    deactivateHoverVideo("inactive");
    activateHoverVideo("next", () => undefined);

    assert.equal(activeStops, 1);
  });
});
