import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { resolveAssetPickerDrop } from "./assetPickerDrop.ts";

const videoFile = new File(["x"], "clip.mp4", { type: "video/mp4" });
const imageFile = new File(["x"], "still.png", { type: "image/png" });

function isVideo(file: File): boolean {
  return file.type.startsWith("video/");
}

describe("resolveAssetPickerDrop", () => {
  it("uses an Explore video asset", () => {
    assert.deepEqual(
      resolveAssetPickerDrop({
        assetPayload: JSON.stringify({ id: "asset-1", media_kind: "video" }),
        file: null,
        mediaKind: "video",
        isEligibleFile: isVideo,
      }),
      { kind: "asset", id: "asset-1" },
    );
  });

  it("uses a project asset whose type is video", () => {
    assert.deepEqual(
      resolveAssetPickerDrop({
        assetPayload: JSON.stringify({ id: "asset-2", type: "video" }),
        file: null,
        mediaKind: "video",
        isEligibleFile: isVideo,
      }),
      { kind: "asset", id: "asset-2" },
    );
  });

  it("ignores an asset of the wrong kind", () => {
    assert.deepEqual(
      resolveAssetPickerDrop({
        assetPayload: JSON.stringify({ id: "asset-3", media_kind: "image" }),
        file: videoFile,
        mediaKind: "video",
        isEligibleFile: isVideo,
      }),
      { kind: "ignore" },
    );
  });

  it("ingests an eligible file when the drag is not an asset", () => {
    assert.deepEqual(
      resolveAssetPickerDrop({
        assetPayload: "",
        file: videoFile,
        mediaKind: "video",
        isEligibleFile: isVideo,
      }),
      { kind: "file", file: videoFile },
    );
  });

  it("ignores an ineligible file", () => {
    assert.deepEqual(
      resolveAssetPickerDrop({
        assetPayload: "not-json",
        file: imageFile,
        mediaKind: "video",
        isEligibleFile: isVideo,
      }),
      { kind: "ignore" },
    );
  });

  it("matches audio the same way", () => {
    const audioFile = new File(["x"], "take.wav", { type: "audio/wav" });
    assert.deepEqual(
      resolveAssetPickerDrop({
        assetPayload: JSON.stringify({ id: "asset-4", type: "audio" }),
        file: audioFile,
        mediaKind: "audio",
        isEligibleFile: (file) => file.type.startsWith("audio/"),
      }),
      { kind: "asset", id: "asset-4" },
    );
  });
});
