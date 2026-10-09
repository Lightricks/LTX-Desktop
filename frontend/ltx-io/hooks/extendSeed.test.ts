import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { EXTEND_SEED_CACHE, planExtendSeed } from "./extendSeed.ts";

const hydratingSpec = {
  params: { prompt: "from history", duration: 4, mode: "end" },
  inputs: { video: { assetId: "history-video" } },
};

describe("planExtendSeed", () => {
  it("skips ingest when persisted extend values already exist", () => {
    assert.equal(
      planExtendSeed({
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
      planExtendSeed({
        hasStoredValues: true,
        generationsReady: true,
        generationsFailed: false,
        lastGenerationSpec: hydratingSpec,
      }),
      "skip",
    );
  });

  it("skips ingest when a durable generation can restore the video", () => {
    assert.equal(
      planExtendSeed({
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
      planExtendSeed({
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
      planExtendSeed({
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
      planExtendSeed({
        hasStoredValues: false,
        generationsReady: false,
        generationsFailed: true,
        lastGenerationSpec: undefined,
      }),
      "skip",
    );
  });
});

describe("extend seed query cache", () => {
  it("retains the seed query forever so remounts and GC do not re-ingest", () => {
    assert.equal(EXTEND_SEED_CACHE.staleTime, Number.POSITIVE_INFINITY);
    assert.equal(EXTEND_SEED_CACHE.gcTime, Number.POSITIVE_INFINITY);
  });
});
