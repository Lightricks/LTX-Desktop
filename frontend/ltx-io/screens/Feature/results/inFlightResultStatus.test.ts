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
      inFlightResultStatus({ status: "cancelling", progressPercent: 40 }),
      {
        title: "Cancellation in progress",
        showDots: false,
      },
    );
  });
});
