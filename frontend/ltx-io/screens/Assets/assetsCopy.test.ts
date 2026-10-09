import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  ASSETS_DELETE_CONFIRM_BUTTON,
  ASSETS_DELETE_CONFIRM_DESCRIPTION,
  ASSETS_DELETE_CONFIRM_TITLE,
  ASSETS_IN_USE_BUTTON,
  ASSETS_IN_USE_DESCRIPTION,
  ASSETS_IN_USE_TITLE,
  ASSETS_EMPTY_CTA,
  ASSETS_EMPTY_DESCRIPTION,
  ASSETS_EMPTY_TITLE,
  ASSETS_ERROR_CTA,
  ASSETS_ERROR_DESCRIPTION,
  ASSETS_ERROR_TITLE,
  ASSETS_FILTERED_EMPTY_CTA,
  ASSETS_FILTERED_EMPTY_TITLE,
  ASSETS_SEARCH_DEBOUNCE_MS,
  ASSETS_SKELETON_COUNT,
} from "./assetsCopy.ts";

describe("assets copy", () => {
  it("sends empty libraries to Home", () => {
    assert.equal(ASSETS_EMPTY_TITLE, "Your library is empty");
    assert.equal(ASSETS_EMPTY_CTA, "Go to Home");
  });

  it("keeps asset deletion confirmation copy exact", () => {
    assert.equal(ASSETS_IN_USE_TITLE, "This file can’t be deleted");
    assert.equal(
      ASSETS_IN_USE_DESCRIPTION,
      "It’s used as an input to another generation.",
    );
    assert.equal(ASSETS_IN_USE_BUTTON, "Got it");
    assert.equal(ASSETS_DELETE_CONFIRM_TITLE, "Delete this file?");
    assert.equal(
      ASSETS_DELETE_CONFIRM_DESCRIPTION,
      "This removes the file from disk and the matching item from Results.",
    );
    assert.equal(ASSETS_DELETE_CONFIRM_BUTTON, "Delete file");
  });

  it("keeps filtered empty, error, loading, and search copy exact", () => {
    assert.equal(
      ASSETS_EMPTY_DESCRIPTION,
      "Videos, images, and audio you generate in Explore are stored here.",
    );
    assert.equal(ASSETS_FILTERED_EMPTY_TITLE, "No media found");
    assert.equal(ASSETS_FILTERED_EMPTY_CTA, "Start again");
    assert.equal(ASSETS_ERROR_TITLE, "Assets aren't loading right now");
    assert.equal(
      ASSETS_ERROR_DESCRIPTION,
      "Something went wrong loading this library. Try again.",
    );
    assert.equal(ASSETS_ERROR_CTA, "Try again");
    assert.equal(ASSETS_SEARCH_DEBOUNCE_MS, 700);
    assert.equal(ASSETS_SKELETON_COUNT, 20);
  });
});
