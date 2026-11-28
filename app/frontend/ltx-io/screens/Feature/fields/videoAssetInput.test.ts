import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { ApiResultError } from "../../../lib/unwrapApiResult.ts";
import {
  COULD_NOT_TRIM_VIDEO_MESSAGE,
  TRIM_BUSY_MESSAGE,
  TRIMMED_VIDEO_TOO_LARGE_MESSAGE,
  videoTrimErrorMessage,
} from "./videoAssetInput.ts";

function apiError(code: string, status: number) {
  return new ApiResultError(code, { code, status });
}

describe("video trim error copy", () => {
  it("asks for a retry while another trim holds the encoder", () => {
    assert.equal(
      videoTrimErrorMessage(apiError("TRIM_BUSY", 503)),
      TRIM_BUSY_MESSAGE,
    );
  });

  it("asks for a shorter range when the output is too large", () => {
    assert.equal(
      videoTrimErrorMessage(apiError("FILE_TOO_LARGE", 400)),
      TRIMMED_VIDEO_TOO_LARGE_MESSAGE,
    );
  });

  it("falls back to the generic message for anything else", () => {
    assert.equal(
      videoTrimErrorMessage(new Error("boom")),
      COULD_NOT_TRIM_VIDEO_MESSAGE,
    );
  });
});
