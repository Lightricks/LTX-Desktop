import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { localUnavailableMessage } from "./local-generation-availability.ts";
import type {
  VideoGenerationModelSpecItem,
  VideoGenerationModelSpecsResponse,
} from "./video-generation-model-specs.ts";

function fastItem(cells: boolean): VideoGenerationModelSpecItem {
  const maps = cells ? { "540p": { fps_to_durations: { "24": [5] } } } : {};
  return {
    pipeline: "fast",
    spec: {
      display_name: "Fast",
      supported_resolutions_durations: maps,
      a2v_supported_resolutions_durations: maps,
    },
  };
}

function specsWith(options: {
  cells: boolean;
  reason?: VideoGenerationModelSpecsResponse["local_unavailable_reason"];
  required?: number | null;
  detected?: number | null;
}): VideoGenerationModelSpecsResponse {
  return {
    api_models: [],
    local_models: [fastItem(options.cells)],
    local_unavailable_reason: options.reason ?? null,
    local_unavailable_required_ram_gb: options.required ?? null,
    local_unavailable_detected_ram_gb: options.detected ?? null,
  };
}

function unavailable(
  specs: VideoGenerationModelSpecsResponse | null,
  platform: string | null = null,
) {
  return localUnavailableMessage(specs, platform);
}

describe("localUnavailableMessage", () => {
  it("returns null while the response itself is missing", () => {
    assert.equal(unavailable(null), null);
  });

  it("returns null when the local grid has cells, even with a reason present", () => {
    assert.equal(
      unavailable(
        specsWith({ cells: true, reason: "ram_shortfall", required: 40, detected: 32 }),
      ),
      null,
    );
  });

  it("returns null for a healthy local grid with no reason", () => {
    assert.equal(unavailable(specsWith({ cells: true })), null);
  });

  it("names required and detected RAM when both are known", () => {
    assert.equal(
      unavailable(
        specsWith({ cells: false, reason: "ram_shortfall", required: 40, detected: 32 }),
      ),
      "Local generation needs 40 GB of system RAM. This machine has 32 GB.",
    );
  });

  it("interpolates the numbers from the response, never literals", () => {
    assert.equal(
      unavailable(
        specsWith({ cells: false, reason: "ram_shortfall", required: 32, detected: 24 }),
      ),
      "Local generation needs 32 GB of system RAM. This machine has 24 GB.",
    );
  });

  it("omits the second sentence when detected RAM is unknown", () => {
    assert.equal(
      unavailable(
        specsWith({
          cells: false,
          reason: "ram_shortfall",
          required: 40,
          detected: null,
        }),
      ),
      "Local generation needs 40 GB of system RAM.",
    );
  });

  it("falls back to the generic line when required RAM is missing", () => {
    assert.equal(
      unavailable(
        specsWith({
          cells: false,
          reason: "ram_shortfall",
          required: null,
          detected: 32,
        }),
      ),
      "This machine can't run any of the available generation settings.",
    );
  });

  it("names the graphics card for stream_over", () => {
    assert.equal(
      unavailable(specsWith({ cells: false, reason: "stream_over" })),
      "This machine's graphics card can't run any of the available settings.",
    );
  });

  it("tells a pool-refused Mac about its allowance, not a graphics card", () => {
    assert.equal(
      unavailable(
        specsWith({ cells: false, reason: "stream_over" }),
        "darwin",
      ),
      "Memory available for video on this Mac can't run any of the available settings.",
    );
  });

  it("names local generation for unsupported", () => {
    assert.equal(
      unavailable(specsWith({ cells: false, reason: "unsupported" })),
      "This machine can't run local generation.",
    );
  });

  it("falls back to the generic line when no reason names the constraint", () => {
    assert.equal(
      unavailable(specsWith({ cells: false, reason: null })),
      "This machine can't run any of the available generation settings.",
    );
  });

  it("treats an absent reason field like a null one (older backends)", () => {
    const specs = specsWith({ cells: false });
    delete specs.local_unavailable_reason;
    delete specs.local_unavailable_required_ram_gb;
    delete specs.local_unavailable_detected_ram_gb;
    assert.equal(
      unavailable(specs),
      "This machine can't run any of the available generation settings.",
    );
  });

  it("degrades unknown future reasons to the generic line, never throws", () => {
    const specs = specsWith({
      cells: false,
      reason: "something_new" as unknown as "ram_shortfall",
    });
    assert.equal(
      unavailable(specs),
      "This machine can't run any of the available generation settings.",
    );
  });

  it("never mentions the API (copy-family rule)", () => {
    const messages = [
      unavailable(
        specsWith({ cells: false, reason: "ram_shortfall", required: 40, detected: 32 }),
      ),
      unavailable(
        specsWith({
          cells: false,
          reason: "ram_shortfall",
          required: 40,
          detected: null,
        }),
      ),
      unavailable(specsWith({ cells: false, reason: "stream_over" })),
      unavailable(
        specsWith({ cells: false, reason: "stream_over" }),
        "darwin",
      ),
      unavailable(specsWith({ cells: false, reason: "unsupported" })),
      unavailable(specsWith({ cells: false, reason: null })),
    ];
    for (const message of messages) {
      assert.ok(message);
      assert.equal(message.toLowerCase().includes("api"), false);
      assert.equal(message.toLowerCase().includes("key"), false);
    }
  });
});
