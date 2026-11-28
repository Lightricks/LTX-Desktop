import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  LTX_INVALID_API_KEY,
  REJECTED_LTX_API_KEY_MESSAGE,
  REJECTED_LTX_API_KEY_REMOTE_MESSAGE,
  UPDATE_LTX_API_KEY_LABEL,
  rejectedLtxApiKeyNotice,
} from "./rejectedLtxApiKey.ts";

describe("rejectedLtxApiKeyNotice", () => {
  it("offers Settings on desktop when the key was rejected and Text encoding is LTX API", () => {
    assert.deepEqual(
      rejectedLtxApiKeyNotice({
        errorCode: LTX_INVALID_API_KEY,
        usesLtxApiTextEncoding: true,
        canOpenSettings: true,
      }),
      {
        body: REJECTED_LTX_API_KEY_MESSAGE,
        openSettingsLabel: UPDATE_LTX_API_KEY_LABEL,
      },
    );
  });

  it("stays quiet on desktop when Text encoding is the local encoder", () => {
    assert.equal(
      rejectedLtxApiKeyNotice({
        errorCode: LTX_INVALID_API_KEY,
        usesLtxApiTextEncoding: false,
        canOpenSettings: true,
      }),
      null,
    );
  });

  it("does not offer Settings when the desktop host forgot the encoding flag", () => {
    assert.equal(
      rejectedLtxApiKeyNotice({
        errorCode: LTX_INVALID_API_KEY,
        usesLtxApiTextEncoding: undefined,
        canOpenSettings: true,
      }),
      null,
    );
  });

  it("explains the same failure on the phone without a Settings button", () => {
    assert.deepEqual(
      rejectedLtxApiKeyNotice({
        errorCode: LTX_INVALID_API_KEY,
        usesLtxApiTextEncoding: undefined,
        canOpenSettings: false,
      }),
      { body: REJECTED_LTX_API_KEY_REMOTE_MESSAGE },
    );
  });

  it("leaves other failure codes alone", () => {
    assert.equal(
      rejectedLtxApiKeyNotice({
        errorCode: "CAPABILITY_FAILED",
        usesLtxApiTextEncoding: true,
        canOpenSettings: true,
      }),
      null,
    );
    assert.equal(
      rejectedLtxApiKeyNotice({
        errorCode: null,
        usesLtxApiTextEncoding: true,
        canOpenSettings: true,
      }),
      null,
    );
  });
});
