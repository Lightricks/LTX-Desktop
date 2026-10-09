import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { mapOsMicrophoneStatus } from "./os-microphone-status.ts";

describe("mapOsMicrophoneStatus", () => {
  it("keeps granted, denied, and not-determined, and maps everything else to unknown", () => {
    assert.equal(mapOsMicrophoneStatus("granted"), "granted");
    assert.equal(mapOsMicrophoneStatus("denied"), "denied");
    assert.equal(mapOsMicrophoneStatus("not-determined"), "not-determined");
    assert.equal(mapOsMicrophoneStatus("restricted"), "unknown");
    assert.equal(mapOsMicrophoneStatus("unknown"), "unknown");
    assert.equal(mapOsMicrophoneStatus(""), "unknown");
  });
});
