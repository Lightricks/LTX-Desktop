import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  AUDIO_ACCEPT,
  AUDIO_EXTENSIONS,
  AUDIO_OR_VIDEO_ACCEPT,
  AUDIO_OR_VIDEO_EXTENSIONS,
} from "../screens/Feature/fields/audioAssetInput.ts";
import {
  VIDEO_ACCEPT,
  VIDEO_EXTENSIONS,
} from "../screens/Feature/fields/videoAssetInput.ts";
import {
  IMAGE_ACCEPT,
  IMAGE_EXTENSIONS,
} from "../screens/Feature/fields/imageAssetInput.ts";
import {
  describeMediaFilter,
  isAcceptedMediaFile,
  requestAcceptsVideo,
  requestExtractsAudioFromVideo,
} from "./mediaInput.ts";

const imageRequest = {
  title: "Start Frame",
  accept: IMAGE_ACCEPT,
  extensions: IMAGE_EXTENSIONS,
  fileInput: null,
};

const audioRequest = {
  title: "Audio",
  accept: AUDIO_ACCEPT,
  extensions: AUDIO_EXTENSIONS,
  fileInput: null,
};

const audioOrVideoRequest = {
  title: "Audio",
  accept: AUDIO_OR_VIDEO_ACCEPT,
  extensions: AUDIO_OR_VIDEO_EXTENSIONS,
  fileInput: null,
};

const videoRequest = {
  title: "Video",
  accept: VIDEO_ACCEPT,
  extensions: VIDEO_EXTENSIONS,
  fileInput: null,
};

describe("isAcceptedMediaFile", () => {
  it("accepts a listed image mime for an image request", () => {
    assert.equal(
      isAcceptedMediaFile(
        { name: "photo.png", type: "image/png" },
        imageRequest,
      ),
      true,
    );
  });

  it("rejects an unlisted video mime for an image request", () => {
    assert.equal(
      isAcceptedMediaFile(
        { name: "clip.mp4", type: "video/mp4" },
        imageRequest,
      ),
      false,
    );
  });

  it("rejects an All-files gif path for an image request", () => {
    assert.equal(
      isAcceptedMediaFile({ name: "/Users/me/loop.gif", type: "" }, imageRequest),
      false,
    );
  });

  it("accepts a listed audio mime for an audio request", () => {
    assert.equal(
      isAcceptedMediaFile(
        { name: "clip.wav", type: "audio/wav" },
        audioRequest,
      ),
      true,
    );
  });

  it("rejects a video mime for an audio-only request", () => {
    assert.equal(
      isAcceptedMediaFile(
        { name: "clip.mp4", type: "video/mp4" },
        audioRequest,
      ),
      false,
    );
  });

  it("falls back to the request extensions when the mime is empty", () => {
    assert.equal(
      isAcceptedMediaFile({ name: "/Users/me/song.wav", type: "" }, audioRequest),
      true,
    );
    assert.equal(
      isAcceptedMediaFile(
        { name: "/Users/me/notes.txt", type: "" },
        audioRequest,
      ),
      false,
    );
  });

  it("accepts a video path for an audio-or-video request", () => {
    assert.equal(
      isAcceptedMediaFile(
        { name: "/Users/me/talk.mp4", type: "" },
        audioOrVideoRequest,
      ),
      true,
    );
  });

  it("strips mime parameters and matches extensions case-insensitively", () => {
    assert.equal(
      isAcceptedMediaFile(
        { name: "clip.wav", type: "audio/wav; codecs=1" },
        audioRequest,
      ),
      true,
    );
    assert.equal(
      isAcceptedMediaFile({ name: "SONG.WAV", type: "" }, audioRequest),
      true,
    );
  });

  it("rejects an extensionless, typeless pick", () => {
    assert.equal(
      isAcceptedMediaFile({ name: "mystery", type: "" }, audioRequest),
      false,
    );
  });
});

describe("requestAcceptsVideo", () => {
  it("is false for image-only and audio-only requests", () => {
    assert.equal(requestAcceptsVideo(imageRequest), false);
    assert.equal(requestAcceptsVideo(audioRequest), false);
  });

  it("is true when the request extensions include a video suffix", () => {
    assert.equal(requestAcceptsVideo(audioOrVideoRequest), true);
    assert.equal(requestAcceptsVideo(videoRequest), true);
  });
});

describe("requestExtractsAudioFromVideo", () => {
  it("is true only for an audio field that also accepts video as a source", () => {
    assert.equal(requestExtractsAudioFromVideo(audioOrVideoRequest), true);
    assert.equal(requestExtractsAudioFromVideo(videoRequest), false);
    assert.equal(requestExtractsAudioFromVideo(audioRequest), false);
    assert.equal(requestExtractsAudioFromVideo(imageRequest), false);
  });
});

describe("describeMediaFilter", () => {
  it("names the native dialog filter after the accepted media", () => {
    assert.equal(describeMediaFilter(imageRequest), "Images");
    assert.equal(describeMediaFilter(audioRequest), "Audio");
    assert.equal(describeMediaFilter(audioOrVideoRequest), "Audio or video");
    assert.equal(describeMediaFilter(videoRequest), "Video");
  });

  it("falls back to a generic name for unknown accept lists", () => {
    assert.equal(
      describeMediaFilter({ accept: "", extensions: [] }),
      "Media",
    );
  });
});
