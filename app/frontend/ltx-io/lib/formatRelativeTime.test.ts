import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { formatRelativeTime } from "./formatRelativeTime.ts";

const NOW = 1_700_000_000_000;
const SECOND = 1000;
const MINUTE = 60 * SECOND;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;
const MONTH = 30 * DAY;
const YEAR = 365 * DAY;

function ago(ms: number): string {
  return formatRelativeTime(NOW - ms, NOW);
}

describe("formatRelativeTime", () => {
  it("reports 'just now' under 45 seconds", () => {
    assert.equal(ago(0), "just now");
    assert.equal(ago(44 * SECOND), "just now");
  });

  it("clamps future timestamps to 'just now'", () => {
    assert.equal(formatRelativeTime(NOW + 10 * MINUTE, NOW), "just now");
  });

  it("rolls up into minutes, hours, days, months, and years", () => {
    assert.equal(ago(MINUTE), "1 minute ago");
    assert.equal(ago(HOUR), "1 hour ago");
    assert.equal(ago(DAY), "1 day ago");
    assert.equal(ago(MONTH), "1 month ago");
    assert.equal(ago(YEAR), "1 year ago");
  });

  it("pluralizes counts above one", () => {
    assert.equal(ago(2 * MINUTE), "2 minutes ago");
    assert.equal(ago(3 * HOUR), "3 hours ago");
    assert.equal(ago(5 * DAY), "5 days ago");
    assert.equal(ago(2 * MONTH), "2 months ago");
    assert.equal(ago(4 * YEAR), "4 years ago");
  });

  it("uses the largest fitting unit at boundaries", () => {
    // 90 minutes reads as hours, not minutes.
    assert.equal(ago(90 * MINUTE), "2 hours ago");
    // 36 hours reads as days.
    assert.equal(ago(36 * HOUR), "2 days ago");
  });
});
