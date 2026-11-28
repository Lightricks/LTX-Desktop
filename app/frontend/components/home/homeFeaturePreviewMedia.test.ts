import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  browserFeaturePreviewCanPlayType,
  canAttachFeaturePreviewSrc,
  featurePreviewCanPlayType,
  HOME_SCROLL_ROOT_ATTR,
  homeMediaObserverRoot,
  previewMimeType,
  shouldPlayFeaturePreview,
  silenceFeaturePreview,
} from "./homeFeaturePreviewMedia.ts";

describe("homeFeaturePreviewMedia", () => {
  it("maps hashed Home preview URLs to a playable MIME type", () => {
    assert.equal(
      previewMimeType("/static/preview-a1b2c3.webm"),
      "video/webm",
    );
    assert.equal(previewMimeType("/static/clip.mp4?v=1"), "video/mp4");
    assert.equal(previewMimeType("/static/poster.webp"), null);
  });

  it("does not attach a WebM preview when the browser cannot decode it", () => {
    assert.equal(
      canAttachFeaturePreviewSrc("preview.webm", () => ""),
      false,
    );
    assert.equal(
      canAttachFeaturePreviewSrc("preview.webm", () => "maybe"),
      true,
    );
    assert.equal(
      canAttachFeaturePreviewSrc("preview.mp4", () => "probably"),
      true,
    );
  });

  it("observes Home media against the layout scroll root", () => {
    assert.equal(HOME_SCROLL_ROOT_ATTR, "data-home-scroll-root");
    const root = {
      closest(selector: string) {
        return selector === `[${HOME_SCROLL_ROOT_ATTR}]` ? this : null;
      },
    };
    const child = {
      closest(selector: string) {
        return root.closest(selector);
      },
    };
    assert.equal(homeMediaObserverRoot(child as unknown as Element), root);
  });

  it("treats a missing capability probe as unsupported", () => {
    assert.equal(featurePreviewCanPlayType("video/webm", null), "");
    assert.equal(browserFeaturePreviewCanPlayType("video/webm"), "");
  });

  it("reads canPlayType from a reused probe without creating DOM", () => {
    const calls: string[] = [];
    const probe = {
      canPlayType(mime: string) {
        calls.push(mime);
        return mime === "video/webm" ? "maybe" : "";
      },
    };

    assert.equal(featurePreviewCanPlayType("video/webm", probe), "maybe");
    assert.equal(featurePreviewCanPlayType("video/mp4", probe), "");
    assert.deepEqual(calls, ["video/webm", "video/mp4"]);
  });

  it("forces muted playback with no volume", () => {
    const video = { muted: false, volume: 1 };
    silenceFeaturePreview(video);
    assert.equal(video.muted, true);
    assert.equal(video.volume, 0);
  });

  it("plays the preview only when motion is allowed and playback has not failed", () => {
    assert.equal(
      shouldPlayFeaturePreview({
        previewFailed: false,
        reducedMotion: false,
        hasBeenVisible: true,
        inView: true,
      }),
      true,
    );
    assert.equal(
      shouldPlayFeaturePreview({
        previewFailed: false,
        reducedMotion: true,
        hasBeenVisible: true,
        inView: true,
      }),
      false,
    );
    assert.equal(
      shouldPlayFeaturePreview({
        previewFailed: true,
        reducedMotion: false,
        hasBeenVisible: true,
        inView: true,
      }),
      false,
    );
    assert.equal(
      shouldPlayFeaturePreview({
        previewFailed: true,
        reducedMotion: true,
        hasBeenVisible: true,
        inView: true,
      }),
      false,
    );
  });

  it("does not play before first visibility or while out of view", () => {
    assert.equal(
      shouldPlayFeaturePreview({
        previewFailed: false,
        reducedMotion: false,
        hasBeenVisible: false,
        inView: false,
      }),
      false,
    );
    assert.equal(
      shouldPlayFeaturePreview({
        previewFailed: false,
        reducedMotion: false,
        hasBeenVisible: true,
        inView: false,
      }),
      false,
    );
  });
});
