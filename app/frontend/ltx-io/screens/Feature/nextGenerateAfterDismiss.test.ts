import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { nextGenerateAfterDismiss } from "./nextGenerateAfterDismiss.ts";

describe("nextGenerateAfterDismiss", () => {
  it("proceeds when the catalog entry is already downloaded", () => {
    assert.equal(
      nextGenerateAfterDismiss(
        [{ id: "painterly", downloaded: true }],
        "painterly",
      ),
      "proceed",
    );
  });

  it("downloads when the catalog entry is missing or not installed", () => {
    assert.equal(nextGenerateAfterDismiss(null, "painterly"), "download");
    assert.equal(nextGenerateAfterDismiss([], "painterly"), "download");
    assert.equal(
      nextGenerateAfterDismiss(
        [{ id: "painterly", downloaded: false }],
        "painterly",
      ),
      "download",
    );
  });
});
