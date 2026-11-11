import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { isSupportedImageAsset } from "./imageAssetInput.ts";

describe("isSupportedImageAsset", () => {
  it("takes a PNG image asset", () => {
    assert.equal(
      isSupportedImageAsset({ media_kind: "image", mime_type: "image/png", name: "a.png" }),
      true,
    );
  });

  it("rejects a GIF cutout stored as an image asset, so no image field gets it", () => {
    assert.equal(
      isSupportedImageAsset({ media_kind: "image", mime_type: "image/gif", name: "cut.gif" }),
      false,
    );
  });

  it("rejects a video asset even when its mime type is an image type", () => {
    assert.equal(
      isSupportedImageAsset({ media_kind: "video", mime_type: "image/png", name: "a.png" }),
      false,
    );
  });
});
