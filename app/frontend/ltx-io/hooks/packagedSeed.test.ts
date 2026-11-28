import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { fromGeneration as fromExtendGeneration } from "../screens/Feature/definitions/extend.ts";
import { fromGeneration as fromRetakeGeneration } from "../screens/Feature/definitions/retake.ts";
import {
  PACKAGED_SEED_CACHE,
  isPackagedSeedName,
  planPackagedSeed,
} from "./packagedSeed.ts";

const RETAKE_SPEC = {
  params: { prompt: "from history", startTime: 0, duration: 2 },
  inputs: { video: { assetId: "history-video" } },
};

const EXTEND_SPEC = {
  params: { prompt: "from history", duration: 4, mode: "end" },
  inputs: { video: { assetId: "history-video" } },
};

const CASES = [
  {
    name: "retake",
    fromGeneration: fromRetakeGeneration,
    spec: RETAKE_SPEC,
    filenames: ["retake-input.mp4", "retake-input-v2.mp4"],
    currentName: "retake-input.mp4",
    previousName: "retake-input-v2.mp4",
  },
  {
    name: "extend",
    fromGeneration: fromExtendGeneration,
    spec: EXTEND_SPEC,
    filenames: ["extend-input-v3.mp4"],
    currentName: "extend-input-v3.mp4",
    previousName: "extend-input-v2.mp4",
  },
] as const;

describe("planPackagedSeed", () => {
  for (const seed of CASES) {
    function plan(
      overrides: Partial<Parameters<typeof planPackagedSeed>[0]> = {},
    ) {
      return planPackagedSeed({
        hasStoredValues: false,
        generationsReady: true,
        generationsFailed: false,
        lastGenerationSpec: undefined,
        fromGeneration: seed.fromGeneration,
        hydratedName: null,
        hydratedPending: false,
        appliedSeedRevision: 1,
        packagedSeedRevision: 1,
        packagedFilenames: seed.filenames,
        ...overrides,
      });
    }

    it(`${seed.name}: skips ingest when persisted values already exist`, () => {
      assert.equal(plan({ hasStoredValues: true, generationsReady: false }), "skip");
    });

    it(`${seed.name}: skips ingest when a durable generation can restore the video`, () => {
      assert.equal(plan({ lastGenerationSpec: seed.spec }), "skip");
    });

    it(`${seed.name}: waits for generations before ingesting on a first visit`, () => {
      assert.equal(plan({ generationsReady: false }), "wait");
    });

    it(`${seed.name}: ingests only when nothing durable can hydrate`, () => {
      assert.equal(plan(), "ingest");
    });

    it(`${seed.name}: does not ingest when generation listing failed`, () => {
      assert.equal(
        plan({ generationsReady: false, generationsFailed: true }),
        "skip",
      );
    });

    it(`${seed.name}: re-ingests a previous packaged example after a revision bump`, () => {
      assert.equal(
        plan({
          lastGenerationSpec: seed.spec,
          hydratedName: seed.previousName,
          appliedSeedRevision: 0,
          packagedSeedRevision: 2,
        }),
        seed.filenames.includes(seed.previousName) ? "ingest" : "skip",
      );
    });

    it(`${seed.name}: keeps a user-uploaded video across seed revisions`, () => {
      assert.equal(
        plan({
          hasStoredValues: true,
          hydratedName: "input enhanced.mp4",
          appliedSeedRevision: 0,
          packagedSeedRevision: 2,
        }),
        "skip",
      );
    });
  }
});

describe("isPackagedSeedName", () => {
  it("matches current and previous packaged filenames", () => {
    assert.equal(
      isPackagedSeedName("retake-input.mp4", ["retake-input.mp4", "retake-input-v2.mp4"]),
      true,
    );
    assert.equal(
      isPackagedSeedName("input enhanced.mp4", ["retake-input.mp4"]),
      false,
    );
    assert.equal(isPackagedSeedName(null, ["retake-input.mp4"]), false);
  });
});

describe("packaged seed query cache", () => {
  it("retains the seed query forever so remounts and GC do not re-ingest", () => {
    assert.equal(PACKAGED_SEED_CACHE.staleTime, Number.POSITIVE_INFINITY);
    assert.equal(PACKAGED_SEED_CACHE.gcTime, Number.POSITIVE_INFINITY);
  });
});
