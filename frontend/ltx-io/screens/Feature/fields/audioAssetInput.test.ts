import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { ApiResultError } from "../../../lib/unwrapApiResult.ts";
import {
  AUDIO_FILE_TOO_LARGE_MESSAGE,
  COULD_NOT_ADD_AUDIO_MESSAGE,
  VIDEO_HAS_NO_AUDIO_MESSAGE,
  audioIngestErrorMessage,
} from "./audioAssetInput.ts";

describe("audio ingest error copy", () => {
  it("says the file is too large for a size failure", () => {
    const oversized = "/tmp/huge.mp4";
    const message = audioIngestErrorMessage(
      new ApiResultError(`file too large: ${oversized}`, {
        code: "FILE_TOO_LARGE",
        status: 400,
      }),
    );

    assert.equal(message, AUDIO_FILE_TOO_LARGE_MESSAGE);
    assert.equal(message.includes(oversized), false);
  });

  it("says a silent video has no audio track", () => {
    const message = audioIngestErrorMessage(
      new ApiResultError("video has no audio track", {
        code: "NO_AUDIO_STREAM",
        status: 400,
      }),
    );

    assert.equal(message, VIDEO_HAS_NO_AUDIO_MESSAGE);
  });

  it("keeps the generic copy for other extract failures", () => {
    const message = audioIngestErrorMessage(
      new ApiResultError("ffmpeg failed to extract audio", {
        code: "UNREADABLE_MEDIA",
        status: 400,
      }),
    );

    assert.equal(message, COULD_NOT_ADD_AUDIO_MESSAGE);
  });
});
