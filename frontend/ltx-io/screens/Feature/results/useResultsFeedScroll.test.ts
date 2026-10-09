import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  isAwayFromNewestResult,
  isStillFocusingResult,
  scrollTopToCenter,
  shouldFollowNewestResult,
} from "./useResultsFeedScroll.ts";

describe("isAwayFromNewestResult", () => {
  it("stays hidden until the newest result leaves the top of the scroller", () => {
    assert.equal(isAwayFromNewestResult(false, 100, 100), false);
    assert.equal(isAwayFromNewestResult(false, 80, 100), false);
    assert.equal(isAwayFromNewestResult(false, 70, 100), true);
  });

  it("stays shown until the newest result is back at the top", () => {
    assert.equal(isAwayFromNewestResult(true, 70, 100), true);
    assert.equal(isAwayFromNewestResult(true, 90, 100), true);
    assert.equal(isAwayFromNewestResult(true, 96, 100), false);
  });
});

describe("isStillFocusingResult", () => {
  it("blocks follow-newest only until that result has been scrolled into view", () => {
    assert.equal(isStillFocusingResult("job-a", null), true);
    assert.equal(isStillFocusingResult("job-a", "job-a"), false);
    assert.equal(isStillFocusingResult("job-b", "job-a"), true);
    assert.equal(isStillFocusingResult(null, "job-a"), false);
  });
});

describe("shouldFollowNewestResult", () => {
  it("does not chase the newest card while a Done item is the scroll target", () => {
    assert.equal(
      shouldFollowNewestResult({
        generationStarted: false,
        pinnedToNewest: true,
        focusingResult: true,
      }),
      false,
    );
  });

  it("follows the newest card again after the Done scroll once the user is pinned", () => {
    assert.equal(
      shouldFollowNewestResult({
        generationStarted: false,
        pinnedToNewest: true,
        focusingResult: false,
      }),
      true,
    );
  });

  it("stays on the Done item after the scroll settles if the user is still away", () => {
    assert.equal(
      shouldFollowNewestResult({
        generationStarted: false,
        pinnedToNewest: false,
        focusingResult: false,
      }),
      false,
    );
  });

  it("still follows the newest card when a generation starts", () => {
    assert.equal(
      shouldFollowNewestResult({
        generationStarted: true,
        pinnedToNewest: true,
        focusingResult: true,
      }),
      true,
    );
  });
});

describe("scrollTopToCenter", () => {
  it("centers a result that sits below the top of the feed", () => {
    assert.equal(
      scrollTopToCenter({
        scrollTop: 0,
        scrollerTop: 0,
        scrollerHeight: 400,
        scrollHeight: 2000,
        targetTop: 900,
        targetHeight: 200,
      }),
      800,
    );
  });
});
