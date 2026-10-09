import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  icLoraResolutionFor,
  icLoraResolutionOffered,
  icLoraResolutionOptions,
} from "./icLoraResolution.ts";

const FHD = { videoWidth: 1920, videoHeight: 1080 };
const HD = { videoWidth: 1280, videoHeight: 720 };
const SMALL = { videoWidth: 320, videoHeight: 240 };
const UNKNOWN = { videoWidth: null, videoHeight: null };

describe("icLoraResolutionFor", () => {
  it("keeps the chosen cell when the source offers it", () => {
    assert.equal(icLoraResolutionFor("1080p", FHD, "720p"), "1080p");
    assert.equal(icLoraResolutionFor("360p", SMALL, "720p"), "360p");
  });

  it("uses the default cell when the source does not offer the chosen one", () => {
    assert.equal(icLoraResolutionFor("1080p", HD, "720p"), "720p");
  });

  it("returns the largest offered cell when the default is not offered either", () => {
    assert.equal(icLoraResolutionFor("1080p", SMALL, "720p"), "540p");
    assert.equal(icLoraResolutionFor("720p", SMALL, "720p"), "540p");
  });

  it("uses the default cell for a value that is not a cell", () => {
    assert.equal(icLoraResolutionFor("4k", FHD, "720p"), "720p");
    assert.equal(icLoraResolutionFor(undefined, FHD, "720p"), "720p");
  });

  it("keeps every cell while the source size is unknown", () => {
    assert.equal(icLoraResolutionFor("1080p", UNKNOWN, "720p"), "1080p");
  });

  it("keeps the chosen cell for a recipe that upscales its source", () => {
    assert.equal(icLoraResolutionFor("1080p", SMALL, "720p", true), "1080p");
    assert.equal(icLoraResolutionFor("4k", SMALL, "720p", true), "720p");
  });
});

describe("icLoraResolutionOffered", () => {
  it("hides 720p and 1080p for a smaller source by default", () => {
    assert.equal(icLoraResolutionOffered("720p", SMALL), false);
    assert.equal(icLoraResolutionOffered("1080p", HD), false);
    assert.equal(icLoraResolutionOffered("1080p", FHD), true);
  });

  it("offers every cell to a recipe that upscales its source", () => {
    assert.equal(icLoraResolutionOffered("720p", SMALL, true), true);
    assert.equal(icLoraResolutionOffered("1080p", SMALL, true), true);
  });
});

describe("icLoraResolutionOptions", () => {
  const values = (options: ReturnType<typeof icLoraResolutionOptions>) =>
    options.map((option) => option.value);

  it("limits the cells to the source size by default", () => {
    assert.deepEqual(values(icLoraResolutionOptions({ ...SMALL, specs: null })), [
      "270p",
      "360p",
      "540p",
    ]);
  });

  it("lists every cell for a small source when the recipe upscales it", () => {
    const options = icLoraResolutionOptions({ ...SMALL, specs: null }, true);
    assert.deepEqual(values(options), ["270p", "360p", "540p", "720p", "1080p"]);
    // A 240p source is below the lowest cell, so no cell is Original.
    assert.deepEqual(
      options.map((option) => option.label),
      ["270p", "360p", "540p", "720p", "1080p"],
    );
  });

  it("tags a cell as Original up to 16 px above the source, as for every cell", () => {
    const original = (videoHeight: number) =>
      icLoraResolutionOptions({ videoWidth: 1920, videoHeight, specs: null }, true).find((option) =>
        option.label.includes("(Original)"),
      )?.value;
    assert.equal(original(254), "270p");
    assert.equal(original(253), undefined);
    assert.equal(original(704), "720p");
    assert.equal(original(703), "540p");
  });
});
