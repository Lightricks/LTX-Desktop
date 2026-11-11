import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { ExploreListedAsset } from "../../../lib/explore-contract.ts";

import {
  formatAspectRatio,
  formatAssetMeta,
  formatClipDuration,
  formatImageMeta,
  formatResolution,
  formatVideoMeta,
} from "./assetMeta.ts";

function videoMetadata(
  durationMs: number,
  width: number,
  height: number,
): ExploreListedAsset["metadata"] {
  return {
    mediaType: "video",
    metadata: { durationMs, width, height, sizeBytes: 1, audioStreamCount: 0 },
  };
}

describe("formatClipDuration", () => {
  it("zero-pads the seconds from one minute up", () => {
    assert.equal(formatClipDuration(60_000), "1:00");
    assert.equal(formatClipDuration(65_000), "1:05");
  });

  it("returns null for a zero, negative or NaN duration, so no '0s' shows", () => {
    assert.equal(formatClipDuration(0), null);
    assert.equal(formatClipDuration(-1), null);
    assert.equal(formatClipDuration(Number.NaN), null);
  });
});

describe("formatAspectRatio", () => {
  it("keeps portrait and landscape apart", () => {
    assert.equal(formatAspectRatio({ width: 1080, height: 1920 }), "9:16");
    assert.equal(formatAspectRatio({ width: 1920, height: 1080 }), "16:9");
  });

  it("snaps a grid size like 1280x704 to 16:9", () => {
    assert.equal(formatAspectRatio({ width: 1280, height: 704 }), "16:9");
  });

  it("returns null outside the tolerance", () => {
    assert.equal(formatAspectRatio({ width: 1000, height: 333 }), null);
  });
});

describe("formatResolution", () => {
  it("uses the short side, so portrait matches landscape", () => {
    assert.equal(formatResolution({ width: 1080, height: 1920 }), "1080p");
  });

  it("snaps grid sizes to the app tiers, including 576 to 540p", () => {
    assert.equal(formatResolution({ width: 1024, height: 576 }), "540p");
    assert.equal(formatResolution({ width: 1280, height: 704 }), "720p");
  });

  it("includes the 270p and 360p tiers", () => {
    assert.equal(formatResolution({ width: 454, height: 256 }), "270p");
    assert.equal(formatResolution({ width: 640, height: 360 }), "360p");
  });

  it("uses the 4K label for 2160", () => {
    assert.equal(formatResolution({ width: 3840, height: 2160 }), "4K");
  });

  it("keeps the own short side when far from every tier, with no false tier", () => {
    assert.equal(formatResolution({ width: 854, height: 480 }), "480p");
    assert.equal(formatResolution({ width: 1920, height: 800 }), "800p");
  });
});

describe("formatVideoMeta", () => {
  it("orders the parts: duration, aspect ratio, resolution", () => {
    assert.deepEqual(formatVideoMeta(videoMetadata(5000, 1920, 1080)), [
      "5s",
      "16:9",
      "1080p",
    ]);
  });

  it("drops an unknown aspect ratio and keeps the other parts in order", () => {
    assert.deepEqual(formatVideoMeta(videoMetadata(5000, 1000, 333)), ["5s", "360p"]);
  });

  it("returns no parts for a non-video asset", () => {
    assert.deepEqual(
      formatVideoMeta({ mediaType: "image", metadata: { width: 10, height: 10 } }),
      [],
    );
  });
});

function imageMetadata(width: number, height: number): ExploreListedAsset["metadata"] {
  return { mediaType: "image", metadata: { width, height } };
}

describe("formatImageMeta", () => {
  it("orders the parts: aspect ratio, resolution", () => {
    assert.deepEqual(formatImageMeta(imageMetadata(1920, 1080)), ["16:9", "1080p"]);
  });

  it("shows the real short side, so 1024x1024 does not read 1080p and 2048 does not read 4K", () => {
    assert.deepEqual(formatImageMeta(imageMetadata(1024, 1024)), ["1:1", "1024p"]);
    assert.deepEqual(formatImageMeta(imageMetadata(2048, 3641)), [
      "9:16",
      "2048p",
    ]);
  });

  it("reads the picker tier for an image the backend rounded down to the 16 px grid", () => {
    // Bug caught: a 1080p pick is stored as 1072 and the badge read "1072p".
    assert.deepEqual(formatImageMeta(imageMetadata(1920, 1072)), ["16:9", "1080p"]);
    assert.deepEqual(formatImageMeta(imageMetadata(1072, 1920)), ["9:16", "1080p"]);
    assert.deepEqual(formatImageMeta(imageMetadata(1072, 1072)), ["1:1", "1080p"]);
    assert.deepEqual(formatImageMeta(imageMetadata(1920, 1056)), ["16:9", "1056p"]);
  });

  it("drops an unknown aspect ratio and keeps the resolution", () => {
    assert.deepEqual(formatImageMeta(imageMetadata(1000, 333)), ["333p"]);
  });
});

describe("formatAssetMeta", () => {
  it("gives audio no badge parts", () => {
    assert.deepEqual(
      formatAssetMeta({ mediaType: "audio", metadata: { durationMs: 1000 } }),
      [],
    );
  });

  it("keeps the video parts for a video", () => {
    assert.deepEqual(formatAssetMeta(videoMetadata(5000, 1920, 1080)), [
      "5s",
      "16:9",
      "1080p",
    ]);
  });
});
