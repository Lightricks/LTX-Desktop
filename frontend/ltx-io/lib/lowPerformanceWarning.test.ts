import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  HIGH_RES_1440P_RUN_WARNING,
  HIGH_RES_RUN_WARNING,
  LOW_PERFORMANCE_RESOLUTION_WARNING,
  chunkedRunNotice,
  generationResolutionWarning,
} from "./lowPerformanceWarning.ts";

describe("chunkedRunNotice", () => {
  it("names the window count", () => {
    assert.match(chunkedRunNotice(3), /3 sections/);
  });

  it("says the time goes up and the result may be less consistent than a single pass", () => {
    const notice = chunkedRunNotice(2);
    assert.match(notice, /takes longer than generating all at once/);
    assert.match(notice, /may be less consistent/);
    assert.match(notice, /drift between sections/);
  });
});

describe("generationResolutionWarning", () => {
  it("keeps the resolution-only warning when the cell is not known", () => {
    assert.equal(generationResolutionWarning("2160p", false), HIGH_RES_RUN_WARNING);
    assert.equal(generationResolutionWarning("1440p", false), HIGH_RES_1440P_RUN_WARNING);
    assert.equal(generationResolutionWarning("1080p", false), undefined);
  });

  it("says a 1080p cell runs in windows, with no resolution hint to add", () => {
    assert.equal(generationResolutionWarning("1080p", false, { windows: 3 }), chunkedRunNotice(3));
  });

  it("shows the window notice on 4K instead of the generic 4K warning", () => {
    assert.equal(generationResolutionWarning("2160p", false, { windows: 4 }), chunkedRunNotice(4));
  });

  it("keeps the 1440p hint after the window notice", () => {
    assert.equal(
      generationResolutionWarning("1440p", false, { windows: 2 }),
      `${chunkedRunNotice(2)} ${HIGH_RES_1440P_RUN_WARNING}`,
    );
  });

  it("does not claim windows for a cell that runs in one pass", () => {
    assert.equal(generationResolutionWarning("2160p", false, null), undefined);
    assert.equal(generationResolutionWarning("1440p", false, null), HIGH_RES_1440P_RUN_WARNING);
  });

  it("says nothing for a 1080p cell in one pass, or the Mac slowness on a slow machine", () => {
    assert.equal(generationResolutionWarning("1080p", false, null), undefined);
    assert.equal(generationResolutionWarning("720p", true, null), LOW_PERFORMANCE_RESOLUTION_WARNING);
  });
});
