import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { describeImageProviders, resolveImageProvider } from "./imageProviderAvailability.ts";

const ready = { defaultProvider: "local", localBlocker: null, falBlocker: null } as const;

function resolve(
  status: Parameters<typeof describeImageProviders>[0],
  picked: "local" | "fal" | null,
) {
  return resolveImageProvider(describeImageProviders(status, "desktop"), status.defaultProvider, picked);
}

describe("image provider availability", () => {
  it("keeps a stale fal pick from blocking Generate after the key is removed", () => {
    const noKey = { ...ready, falBlocker: "FAL_API_KEY_NOT_CONFIGURED" } as const;
    assert.equal(resolve(noKey, "fal"), "local");
  });

  it("sends a machine that cannot run locally to fal even when the user picked local", () => {
    const forcedApi = {
      defaultProvider: "fal",
      localBlocker: "LOCAL_GENERATION_UNSUPPORTED",
      falBlocker: null,
    } as const;
    assert.equal(resolve(forcedApi, "local"), "fal");
  });

  it("moves to fal while the model is missing and keeps the download message for local", () => {
    const missing = { ...ready, localBlocker: "IMAGE_MODEL_NOT_DOWNLOADED" } as const;
    assert.equal(resolve(missing, null), "fal");
    const local = describeImageProviders(missing, "desktop")[0];
    assert.equal(local?.blockedReason, "Download the Z-Image Turbo model in Settings.");
  });

  it("names Desktop as the place to fix a blocker on Remote, which has no Settings", () => {
    const noKey = { ...ready, falBlocker: "FAL_API_KEY_NOT_CONFIGURED" } as const;
    const fal = describeImageProviders(noKey, "remote")[1];
    assert.equal(fal?.blockedReason, "Add a fal API key on Desktop.");
  });

  it("returns the default provider when neither can run, so Generate shows its blocker", () => {
    const none = {
      defaultProvider: "fal",
      localBlocker: "LOCAL_GENERATION_UNSUPPORTED",
      falBlocker: "FAL_API_KEY_NOT_CONFIGURED",
    } as const;
    assert.equal(resolve(none, "local"), "fal");
  });
});
