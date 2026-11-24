import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  isLightboxGallerySwipePointer,
  swipeGalleryDirection,
} from "./lightboxSwipe.ts";

describe("swipeGalleryDirection", () => {
  it("treats a left swipe as next", () => {
    assert.equal(swipeGalleryDirection(200, 40, 80, 44), "next");
  });

  it("treats a right swipe as previous", () => {
    assert.equal(swipeGalleryDirection(80, 40, 200, 36), "previous");
  });

  it("ignores short or mostly vertical gestures", () => {
    assert.equal(swipeGalleryDirection(100, 40, 130, 42), null);
    assert.equal(swipeGalleryDirection(100, 40, 40, 180), null);
  });
});

describe("lightbox swipe pointers", () => {
  it("treats touch and pen as gallery swipes", () => {
    assert.equal(isLightboxGallerySwipePointer("touch"), true);
    assert.equal(isLightboxGallerySwipePointer("pen"), true);
    assert.equal(isLightboxGallerySwipePointer("mouse"), false);
  });
});
