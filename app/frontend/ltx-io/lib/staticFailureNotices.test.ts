import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { BET_FAILURE_BACKSTOP } from "../../lib/budget-risk.ts";
import {
  CUDA_OOM,
  CUDA_OOM_MESSAGE,
  JOB_TOO_LARGE,
  JOB_TOO_LARGE_MESSAGE,
  staticFailureNotice,
} from "./staticFailureNotices.ts";

describe("staticFailureNotice", () => {
  it("explains a mid-run out-of-memory instead of showing the raw code", () => {
    assert.deepEqual(staticFailureNotice({ errorCode: CUDA_OOM }), {
      body: CUDA_OOM_MESSAGE,
    });
  });

  it("explains an execute-time budget miss instead of showing the raw code", () => {
    assert.deepEqual(staticFailureNotice({ errorCode: JOB_TOO_LARGE }), {
      body: JOB_TOO_LARGE_MESSAGE,
    });
  });

  it("appends the bet backstop only when the failed job was a bet", () => {
    assert.equal(
      staticFailureNotice({ errorCode: CUDA_OOM, ranOnBet: true })?.body,
      `${CUDA_OOM_MESSAGE} ${BET_FAILURE_BACKSTOP}`,
    );
    assert.equal(
      staticFailureNotice({ errorCode: JOB_TOO_LARGE, ranOnBet: false })?.body,
      JOB_TOO_LARGE_MESSAGE,
    );
  });

  it("leaves other failure codes alone", () => {
    assert.equal(staticFailureNotice({ errorCode: "EXECUTOR_FAILED" }), null);
    assert.equal(staticFailureNotice({ errorCode: null }), null);
    assert.equal(staticFailureNotice({ errorCode: undefined }), null);
  });
});
