import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  EXPLORE_IMAGE_TO_VIDEO_SEED_CACHE,
  planExploreImageToVideoSeed,
} from "./exploreImageToVideoSeed.ts";

const hydratingSpec = {
  params: { prompt: "from history", aspectRatio: "auto" },
  inputs: { startFrame: { assetId: "history-start" } },
};

describe("planExploreImageToVideoSeed", () => {
  it("skips ingest when persisted I2V values already exist", () => {
    assert.equal(
      planExploreImageToVideoSeed({
        hasStoredValues: true,
        generationsReady: false,
        generationsFailed: false,
        lastGenerationSpec: undefined,
      }),
      "skip",
    );
  });

  it("skips ingest after an explicit persisted clear", () => {
    assert.equal(
      planExploreImageToVideoSeed({
        hasStoredValues: true,
        generationsReady: true,
        generationsFailed: false,
        lastGenerationSpec: hydratingSpec,
      }),
      "skip",
    );
  });

  it("skips ingest when a durable generation can restore Start Frame", () => {
    assert.equal(
      planExploreImageToVideoSeed({
        hasStoredValues: false,
        generationsReady: true,
        generationsFailed: false,
        lastGenerationSpec: hydratingSpec,
      }),
      "skip",
    );
  });

  it("waits for generations before ingesting on a first visit", () => {
    assert.equal(
      planExploreImageToVideoSeed({
        hasStoredValues: false,
        generationsReady: false,
        generationsFailed: false,
        lastGenerationSpec: undefined,
      }),
      "wait",
    );
  });

  it("ingests the packaged seed only when nothing durable can hydrate", () => {
    assert.equal(
      planExploreImageToVideoSeed({
        hasStoredValues: false,
        generationsReady: true,
        generationsFailed: false,
        lastGenerationSpec: undefined,
      }),
      "ingest",
    );
  });

  it("does not ingest when generation listing failed", () => {
    assert.equal(
      planExploreImageToVideoSeed({
        hasStoredValues: false,
        generationsReady: false,
        generationsFailed: true,
        lastGenerationSpec: undefined,
      }),
      "skip",
    );
  });
});

describe("explore I2V seed query cache", () => {
  it("retains the seed query forever so remounts and GC do not re-ingest", () => {
    assert.equal(EXPLORE_IMAGE_TO_VIDEO_SEED_CACHE.staleTime, Number.POSITIVE_INFINITY);
    assert.equal(EXPLORE_IMAGE_TO_VIDEO_SEED_CACHE.gcTime, Number.POSITIVE_INFINITY);
  });
});
