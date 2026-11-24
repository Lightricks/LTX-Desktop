import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { ApiResultError, unwrapApiResult } from "./unwrapApiResult.ts";

describe("unwrapApiResult", () => {
  it("returns data when ok is true", () => {
    assert.equal(unwrapApiResult({ ok: true, data: { id: "gen-1" } }).id, "gen-1");
  });

  it("throws when ok is false", () => {
    assert.throws(
      () =>
        unwrapApiResult({
          ok: false,
          error: { message: "GENERATION_NOT_FOUND" },
        }),
      { message: "GENERATION_NOT_FOUND" },
    );
  });

  it("preserves the error code and status on the thrown error", () => {
    try {
      unwrapApiResult({
        ok: false,
        status: 402,
        error: { message: "Out of credits", code: "INSUFFICIENT_CREDITS" },
      });
      assert.fail("expected unwrapApiResult to throw");
    } catch (error) {
      assert.ok(error instanceof ApiResultError);
      assert.equal(error.message, "Out of credits");
      assert.equal(error.code, "INSUFFICIENT_CREDITS");
      assert.equal(error.status, 402);
    }
  });

  it("leaves code and status undefined when absent", () => {
    try {
      unwrapApiResult({ ok: false, error: { message: "boom" } });
      assert.fail("expected unwrapApiResult to throw");
    } catch (error) {
      assert.ok(error instanceof ApiResultError);
      assert.equal(error.code, undefined);
      assert.equal(error.status, undefined);
    }
  });
});
