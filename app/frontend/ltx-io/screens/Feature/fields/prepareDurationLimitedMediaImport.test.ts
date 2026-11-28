import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { CloseReason } from "../../../components/shared/Modal/modalTypes.ts";
import type { ExploreAsset } from "../../../../lib/explore-contract.ts";
import { createTrimModalSave } from "../trim/trimModalSave.ts";
import {
  audioDurationSeconds,
  prepareDurationLimitedMediaImport,
  type MediaTrimRequest,
} from "./prepareDurationLimitedMediaImport.ts";

function audioAsset(id: string, durationMs: number): ExploreAsset {
  return {
    id,
    media_kind: "audio",
    origin: "uploaded",
    mime_type: "audio/wav",
    name: `${id}.wav`,
    created_at: 0,
    path: `/tmp/${id}.wav`,
    metadata: { mediaType: "audio", metadata: { durationMs } },
  };
}

function videoAsset(id: string, durationMs: number): ExploreAsset {
  return {
    id,
    media_kind: "video",
    origin: "uploaded",
    mime_type: "video/mp4",
    name: `${id}.mp4`,
    created_at: 0,
    path: `/tmp/${id}.mp4`,
    metadata: {
      mediaType: "video",
      metadata: {
        durationMs,
        width: 1920,
        height: 1080,
        sizeBytes: 1,
        audioStreamCount: 1,
      },
    },
  };
}

type Recorder = {
  accepted: ExploreAsset[];
  trimCalls: { assetId: string; startSec: number; endSec: number }[];
  trimRequests: MediaTrimRequest[];
};

function recorder(): Recorder {
  return { accepted: [], trimCalls: [], trimRequests: [] };
}

function runImport(
  log: Recorder,
  input: {
    asset: ExploreAsset;
    trimCapSeconds: number | null;
    toleranceSeconds?: number;
    openTrim?: (request: MediaTrimRequest) => void;
    trim?: (
      assetId: string,
      startSec: number,
      endSec: number,
    ) => Promise<ExploreAsset>;
  },
) {
  return prepareDurationLimitedMediaImport({
    asset: input.asset,
    trimCapSeconds: input.trimCapSeconds,
    toleranceSeconds: input.toleranceSeconds,
    trim: async (assetId, startSec, endSec) => {
      log.trimCalls.push({ assetId, startSec, endSec });
      if (input.trim) return input.trim(assetId, startSec, endSec);
      return audioAsset(`${assetId}-trimmed`, (endSec - startSec) * 1000);
    },
    onAccept: (asset) => {
      log.accepted.push(asset);
    },
    openTrim: (request) => {
      log.trimRequests.push(request);
      input.openTrim?.(request);
    },
  });
}

describe("prepareDurationLimitedMediaImport", () => {
  it("writes the asset when durationMs/1000 <= cap", async () => {
    const log = recorder();
    const asset = audioAsset("a1", 10_000);

    const outcome = await runImport(log, { asset, trimCapSeconds: 10 });

    assert.deepEqual(outcome, { status: "accepted", asset });
    assert.deepEqual(log.accepted, [asset]);
    assert.equal(log.trimRequests.length, 0);
    assert.equal(log.trimCalls.length, 0);
  });

  it("opens trim and does not write when duration exceeds cap", async () => {
    const log = recorder();
    const asset = audioAsset("a2", 31_000);

    const outcome = await runImport(log, { asset, trimCapSeconds: 30 });

    assert.deepEqual(outcome, { status: "trimming" });
    assert.deepEqual(log.accepted, []);
    assert.equal(log.trimCalls.length, 0);
    assert.equal(log.trimRequests.length, 1);
    assert.equal(log.trimRequests[0].asset, asset);
    assert.equal(log.trimRequests[0].capSeconds, 30);
    assert.equal(log.trimRequests[0].durationSeconds, 31);
  });

  it("does not write when Trim is cancelled", async () => {
    const log = recorder();
    const asset = audioAsset("a4", 60_000);

    const outcome = await runImport(log, {
      asset,
      trimCapSeconds: 20,
      // Cancelling means the modal closes without ever calling onSave.
      openTrim: () => {},
    });

    assert.deepEqual(outcome, { status: "trimming" });
    assert.deepEqual(log.accepted, []);
    assert.equal(log.trimCalls.length, 0);
  });

  it("skips auto-trim when trimCapSeconds is null", async () => {
    const log = recorder();
    const asset = audioAsset("a5", 600_000);

    const outcome = await runImport(log, { asset, trimCapSeconds: null });

    assert.deepEqual(outcome, { status: "accepted", asset });
    assert.deepEqual(log.accepted, [asset]);
    assert.equal(log.trimRequests.length, 0);
  });

  it("accepts a video at the cap without trimming", async () => {
    const log = recorder();
    const asset = videoAsset("v1", 60_000);

    const outcome = await runImport(log, { asset, trimCapSeconds: 60 });

    assert.deepEqual(outcome, { status: "accepted", asset });
    assert.equal(log.trimRequests.length, 0);
  });

  it("opens trim for a video over the cap and accepts the trimmed clip", async () => {
    const log = recorder();
    const asset = videoAsset("v2", 95_000);
    const trimmed = videoAsset("v2-clip", 10_000);

    const outcome = await runImport(log, {
      asset,
      trimCapSeconds: 10,
      trim: async () => trimmed,
    });
    assert.deepEqual(outcome, { status: "trimming" });
    assert.equal(log.trimRequests[0].durationSeconds, 95);
    assert.equal(log.trimRequests[0].capSeconds, 10);

    await log.trimRequests[0].onSave(5, 15);
    assert.deepEqual(log.trimCalls, [{ assetId: "v2", startSec: 5, endSec: 15 }]);
    assert.deepEqual(log.accepted, [trimmed]);
  });
});

/**
 * The modal's Done button runs `createTrimModalSave(request, hideModal).save`.
 * These exercise that binding rather than calling `request.onSave` directly, so
 * a Done button that grew its own copy of trim-then-accept fails here.
 */
describe("trim modal Done (shipped path)", () => {
  it("trims and accepts through the import request, then closes with apply", async () => {
    const log = recorder();
    const asset = audioAsset("d1", 60_000);
    const trimmed = audioAsset("d1-clip", 8_000);
    const closed: CloseReason[] = [];

    await runImport(log, {
      asset,
      trimCapSeconds: 20,
      trim: async () => trimmed,
    });
    const done = createTrimModalSave(log.trimRequests[0], (reason) =>
      closed.push(reason),
    );

    await done.save(2, 10);

    assert.deepEqual(log.trimCalls, [
      { assetId: "d1", startSec: 2, endSec: 10 },
    ]);
    assert.deepEqual(log.accepted, [trimmed]);
    assert.deepEqual(closed, ["apply"]);
    assert.equal(done.isSaving(), false);
  });

  it("reports a failed trim to the modal and keeps it open", async () => {
    const log = recorder();
    const closed: CloseReason[] = [];

    await runImport(log, {
      asset: audioAsset("d2", 60_000),
      trimCapSeconds: 20,
      trim: async () => {
        throw new Error("TRIM_FAILED");
      },
    });
    const done = createTrimModalSave(log.trimRequests[0], (reason) =>
      closed.push(reason),
    );

    // Rejecting is how the modal learns to render its error slot; swallowing
    // here would leave the user staring at a silent overlay.
    await assert.rejects(() => done.save(0, 10), /TRIM_FAILED/);
    assert.deepEqual(log.accepted, []);
    assert.deepEqual(closed, []);
    assert.equal(done.isSaving(), false);
  });

  it("joins a second Done to the trim already in flight", async () => {
    const log = recorder();
    const closed: CloseReason[] = [];
    let releaseTrim: (() => void) | null = null;

    await runImport(log, {
      asset: audioAsset("d3", 60_000),
      trimCapSeconds: 20,
      trim: async (assetId, startSec, endSec) => {
        await new Promise<void>((resolve) => {
          releaseTrim = resolve;
        });
        return audioAsset(`${assetId}-clip`, (endSec - startSec) * 1000);
      },
    });
    const done = createTrimModalSave(log.trimRequests[0], (reason) =>
      closed.push(reason),
    );

    const first = done.save(0, 10);
    assert.equal(done.isSaving(), true);
    const second = done.save(0, 10);
    assert.equal(second, first);
    releaseTrim?.();
    await Promise.all([first, second]);

    assert.equal(log.trimCalls.length, 1);
    assert.deepEqual(closed, ["apply"]);
  });
});

describe("duration cap tolerance", () => {
  it("trims audio just over the cap: a2v frame counts follow its length", async () => {
    const log = recorder();
    const outcome = await runImport(log, {
      asset: audioAsset("a", 20_050),
      trimCapSeconds: 20,
    });
    assert.equal(outcome.status, "trimming");
  });

  it("accepts a video within the tolerance so a 60.03s clip skips the modal", async () => {
    const log = recorder();
    const asset = videoAsset("v", 60_030);
    const outcome = await runImport(log, {
      asset,
      trimCapSeconds: 60,
      toleranceSeconds: 0.1,
    });
    assert.deepEqual(outcome, { status: "accepted", asset });
  });

  it("trims a video past the tolerance", async () => {
    const log = recorder();
    const outcome = await runImport(log, {
      asset: videoAsset("v", 60_200),
      trimCapSeconds: 60,
      toleranceSeconds: 0.1,
    });
    assert.equal(outcome.status, "trimming");
  });
});

describe("audioDurationSeconds", () => {
  it("reads durationMs from audio metadata", () => {
    assert.equal(audioDurationSeconds(audioAsset("a", 2_500)), 2.5);
  });

  it("reads durationMs from video metadata for a video source", () => {
    assert.equal(audioDurationSeconds(videoAsset("v", 7_000)), 7);
  });

  it("returns null when metadata carries no duration", () => {
    const imageLike: ExploreAsset = {
      id: "i",
      media_kind: "image",
      origin: "uploaded",
      mime_type: "image/png",
      name: "i.png",
      created_at: 0,
      path: "/tmp/i.png",
      metadata: { mediaType: "image", metadata: { width: 2, height: 2 } },
    };

    assert.equal(audioDurationSeconds(imageLike), null);
  });
});
