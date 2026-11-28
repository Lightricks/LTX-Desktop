import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { resolveAssetLibraryFileAction } from "./assetLibraryFileAction.ts";

const revealInFolder = () => undefined;
const mediaUrl = "https://example.com/clip.mp4";
const filePath = "/tmp/clip.mp4";

describe("resolveAssetLibraryFileAction", () => {
  it("prefers reveal over download", () => {
    assert.deepEqual(
      resolveAssetLibraryFileAction({
        revealInFolder,
        path: filePath,
        mediaUrl,
      }),
      { kind: "reveal", path: filePath },
    );
  });

  it("downloads when reveal is unavailable and a media url exists", () => {
    assert.deepEqual(
      resolveAssetLibraryFileAction({
        revealInFolder: null,
        path: filePath,
        mediaUrl,
      }),
      { kind: "download", url: mediaUrl },
    );
  });

  it("downloads when the path is empty", () => {
    assert.deepEqual(
      resolveAssetLibraryFileAction({
        revealInFolder,
        path: "",
        mediaUrl,
      }),
      { kind: "download", url: mediaUrl },
    );
  });

  it("returns none when reveal and download are both missing", () => {
    assert.deepEqual(
      resolveAssetLibraryFileAction({
        revealInFolder: null,
        path: null,
        mediaUrl: null,
      }),
      { kind: "none" },
    );
  });
});
