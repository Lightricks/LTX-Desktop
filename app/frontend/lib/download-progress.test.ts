import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { downloadProgressDetail } from "./download-progress.ts";

describe("downloadProgressDetail", () => {
  it("renders bytes and ETA while downloading", () => {
    assert.equal(
      downloadProgressDetail({
        statusLabel: "Downloading Python",
        percent: 42,
        downloadedBytes: 1024 * 1024 * 100,
        totalBytes: 1024 * 1024 * 200,
        speedBytesPerSec: (1024 * 1024 * 100) / 90,
      }),
      "100.0 MB / 200.0 MB · ETA: 2m",
    );
  });

  it("omits the ETA once nothing remains", () => {
    assert.equal(
      downloadProgressDetail({
        statusLabel: "Installed",
        percent: 100,
        downloadedBytes: 200,
        totalBytes: 200,
        speedBytesPerSec: 10,
      }),
      "200.0 B / 200.0 B",
    );
  });

  it("returns null when there is nothing to show", () => {
    assert.equal(
      downloadProgressDetail({ statusLabel: "Downloading", percent: 0 }),
      null,
    );
  });
});
