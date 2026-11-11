import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { inFlightResultStatus } from "./inFlightResultStatus.ts";

describe("inFlightResultStatus", () => {
  it("shows dots while the active generation has no percent yet", () => {
    assert.deepEqual(
      inFlightResultStatus({ status: "running", progressPercent: undefined }),
      {
        showDots: true,
      },
    );
  });

  it("replaces the dots with a rounded percent once progress arrives", () => {
    assert.deepEqual(inFlightResultStatus({ status: "running", progressPercent: 42.2 }), {
      title: "42%",
      showDots: false,
    });
  });

  it("keeps cancellation copy instead of a percent", () => {
    assert.deepEqual(
      inFlightResultStatus({ status: "cancelling", progressPercent: 40, remainingSeconds: 90 }),
      {
        title: "Cancellation in progress",
        showDots: false,
      },
    );
  });

  it("shows remaining time body while running and omits it when cancelling", () => {
    assert.deepEqual(
      inFlightResultStatus({ status: "running", progressPercent: 50, remainingSeconds: 120 }),
      {
        title: "50%",
        body: "About 2 min left",
        showDots: false,
      },
    );
  });

  it("shows Finishing when remainingSeconds reaches 0", () => {
    assert.deepEqual(
      inFlightResultStatus({ status: "running", progressPercent: 95, remainingSeconds: 0 }),
      {
        title: "95%",
        body: "Finishing…",
        showDots: false,
      },
    );
  });
});
