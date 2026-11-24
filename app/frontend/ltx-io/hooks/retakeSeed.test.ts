import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  RETAKE_SEED_CACHE,
  RETAKE_SEED_REVISION,
  isPackagedRetakeSeedName,
  planRetakeSeed,
} from "./retakeSeed.ts";

const hydratingSpec = {
  params: { prompt: "from history", startTime: 0, duration: 2 },
  inputs: { video: { assetId: "history-video" } },
};

function plan(
  overrides: Partial<Parameters<typeof planRetakeSeed>[0]> = {},
) {
  return planRetakeSeed({
    hasStoredValues: false,
    generationsReady: true,
    generationsFailed: false,
    lastGenerationSpec: undefined,
    hydratedVideoName: null,
    hydratedVideoPending: false,
    appliedSeedRevision: RETAKE_SEED_REVISION,
    packagedSeedRevision: RETAKE_SEED_REVISION,
    ...overrides,
  });
}

describe("planRetakeSeed", () => {
  it("skips ingest when persisted retake values already exist", () => {
    assert.equal(plan({ hasStoredValues: true, generationsReady: false }), "skip");
  });

  it("skips ingest after an explicit persisted clear", () => {
    assert.equal(
      plan({ hasStoredValues: true, lastGenerationSpec: hydratingSpec }),
      "skip",
    );
  });

  it("does not re-seed a persisted clear from generation history", () => {
    assert.equal(
      plan({
        hasStoredValues: true,
        lastGenerationSpec: hydratingSpec,
        hydratedVideoName: null,
        appliedSeedRevision: 0,
        packagedSeedRevision: 2,
      }),
      "skip",
    );
  });

  it("skips ingest when a durable generation can restore the video", () => {
    assert.equal(plan({ lastGenerationSpec: hydratingSpec }), "skip");
  });

  it("waits for generations before ingesting on a first visit", () => {
    assert.equal(plan({ generationsReady: false }), "wait");
  });

  it("ingests the packaged seed only when nothing durable can hydrate", () => {
    assert.equal(plan(), "ingest");
  });

  it("does not ingest when generation listing failed", () => {
    assert.equal(
      plan({ generationsReady: false, generationsFailed: true }),
      "skip",
    );
  });

  it("waits while a hydrated example video is still loading", () => {
    assert.equal(
      plan({
        lastGenerationSpec: hydratingSpec,
        hydratedVideoPending: true,
        appliedSeedRevision: 0,
      }),
      "wait",
    );
  });

  it("does not block the form once this seed revision is already applied", () => {
    assert.equal(
      plan({
        lastGenerationSpec: hydratingSpec,
        hydratedVideoPending: true,
      }),
      "skip",
    );
  });

  it("re-ingests when the hydrated video is the previous packaged example", () => {
    assert.equal(
      plan({
        lastGenerationSpec: hydratingSpec,
        hydratedVideoName: "retake-input-v2.mp4",
        appliedSeedRevision: 0,
      }),
      "ingest",
    );
  });

  it("re-ingests when the hydrated video is the current example at an old revision", () => {
    assert.equal(
      plan({
        hasStoredValues: true,
        hydratedVideoName: "retake-input.mp4",
        appliedSeedRevision: 1,
        packagedSeedRevision: 2,
      }),
      "ingest",
    );
  });

  it("keeps a user-uploaded video even if the packaged seed revision changed", () => {
    assert.equal(
      plan({
        hasStoredValues: true,
        hydratedVideoName: "input enhanced.mp4",
        appliedSeedRevision: 0,
        packagedSeedRevision: 2,
      }),
      "skip",
    );
  });

  it("skips a packaged example that already matches this revision", () => {
    assert.equal(
      plan({
        lastGenerationSpec: hydratingSpec,
        hydratedVideoName: "retake-input.mp4",
      }),
      "skip",
    );
  });
});

describe("isPackagedRetakeSeedName", () => {
  it("matches the current and previous packaged filenames", () => {
    assert.equal(isPackagedRetakeSeedName("retake-input.mp4"), true);
    assert.equal(isPackagedRetakeSeedName("retake-input-v2.mp4"), true);
    assert.equal(isPackagedRetakeSeedName("input enhanced.mp4"), false);
    assert.equal(isPackagedRetakeSeedName(null), false);
  });
});

describe("retake seed query cache", () => {
  it("retains the seed query forever so remounts and GC do not re-ingest", () => {
    assert.equal(RETAKE_SEED_CACHE.staleTime, Number.POSITIVE_INFINITY);
    assert.equal(RETAKE_SEED_CACHE.gcTime, Number.POSITIVE_INFINITY);
  });
});
