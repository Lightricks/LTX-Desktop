import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { planExploreAudioToVideoSeed } from "./exploreAudioToVideoSeed.ts";

const hydratingSpec = {
  params: { prompt: "from history", aspectRatio: "auto" },
  inputs: {
    audio: { assetId: "history-audio" },
    startFrame: { assetId: "history-start" },
  },
};

/** A2V hydrates off `inputs.audio`; a start frame alone is not restorable. */
const startFrameOnlySpec = {
  params: { prompt: "from history", aspectRatio: "auto" },
  inputs: { startFrame: { assetId: "history-start" } },
};

describe("planExploreAudioToVideoSeed", () => {
  it("skips ingest when persisted A2V values already exist", () => {
    assert.equal(
      planExploreAudioToVideoSeed({
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
      planExploreAudioToVideoSeed({
        hasStoredValues: true,
        generationsReady: true,
        generationsFailed: false,
        lastGenerationSpec: hydratingSpec,
      }),
      "skip",
    );
  });

  it("skips ingest when a durable generation can restore the audio slot", () => {
    assert.equal(
      planExploreAudioToVideoSeed({
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
      planExploreAudioToVideoSeed({
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
      planExploreAudioToVideoSeed({
        hasStoredValues: false,
        generationsReady: true,
        generationsFailed: false,
        lastGenerationSpec: undefined,
      }),
      "ingest",
    );
  });

  it("ingests when the last generation has no audio to restore", () => {
    assert.equal(
      planExploreAudioToVideoSeed({
        hasStoredValues: false,
        generationsReady: true,
        generationsFailed: false,
        lastGenerationSpec: startFrameOnlySpec,
      }),
      "ingest",
    );
  });

  it("does not ingest when generation listing failed", () => {
    assert.equal(
      planExploreAudioToVideoSeed({
        hasStoredValues: false,
        generationsReady: false,
        generationsFailed: true,
        lastGenerationSpec: undefined,
      }),
      "skip",
    );
  });
});
