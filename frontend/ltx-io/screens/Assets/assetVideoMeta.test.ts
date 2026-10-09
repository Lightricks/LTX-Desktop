import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { ExploreListedAsset } from "../../../lib/explore-contract.ts";

import {
  formatAspectRatio,
  formatClipDuration,
  formatResolution,
  formatVideoMeta,
} from "./assetVideoMeta.ts";

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
