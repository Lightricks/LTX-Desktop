import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  LTX_API_PROMPT_EMBEDDING_FAILED,
  OPEN_TEXT_ENCODING_LABEL,
  PROMPT_EMBEDDING_FAILED_MESSAGE,
  PROMPT_EMBEDDING_FAILED_REMOTE_MESSAGE,
  promptEmbeddingFailureNotice,
} from "./promptEmbeddingFailure.ts";

describe("promptEmbeddingFailureNotice", () => {
  it("offers Text encoding on desktop when API encoding failed and is still selected", () => {
    assert.deepEqual(
      promptEmbeddingFailureNotice({
        errorCode: LTX_API_PROMPT_EMBEDDING_FAILED,
        usesLtxApiTextEncoding: true,
        canOpenSettings: true,
      }),
      {
        body: PROMPT_EMBEDDING_FAILED_MESSAGE,
        openSettingsLabel: OPEN_TEXT_ENCODING_LABEL,
      },
    );
  });

  it("stays quiet on desktop when Text encoding is the local encoder", () => {
    assert.equal(
      promptEmbeddingFailureNotice({
        errorCode: LTX_API_PROMPT_EMBEDDING_FAILED,
        usesLtxApiTextEncoding: false,
        canOpenSettings: true,
      }),
      null,
    );
  });

  it("does not offer Settings when the desktop host forgot the encoding flag", () => {
    assert.equal(
      promptEmbeddingFailureNotice({
        errorCode: LTX_API_PROMPT_EMBEDDING_FAILED,
        usesLtxApiTextEncoding: undefined,
        canOpenSettings: true,
      }),
      null,
    );
  });

  it("explains the same failure on the phone without a Settings button", () => {
    assert.deepEqual(
      promptEmbeddingFailureNotice({
        errorCode: LTX_API_PROMPT_EMBEDDING_FAILED,
        usesLtxApiTextEncoding: undefined,
        canOpenSettings: false,
      }),
      { body: PROMPT_EMBEDDING_FAILED_REMOTE_MESSAGE },
    );
  });

  it("leaves other failure codes alone", () => {
    assert.equal(
      promptEmbeddingFailureNotice({
        errorCode: "CAPABILITY_FAILED",
        usesLtxApiTextEncoding: true,
        canOpenSettings: true,
      }),
      null,
    );
  });
});
