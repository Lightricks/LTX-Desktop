import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { pairingErrorMessage } from "./remote-pairing-screen.ts";

const disabled = {
  permitted: true,
  serving: false,
  reason: "disabled in Settings",
};

describe("pairingErrorMessage", () => {
  it("does not show disabled-in-Settings while Off", () => {
    assert.equal(pairingErrorMessage(false, false, disabled, false), null);
  });

  it("treats stale disabled-in-Settings as startup after clicking On", () => {
    assert.equal(pairingErrorMessage(true, false, disabled, false), null);
  });

  it("treats controller startup reasons as loading, not errors", () => {
    for (const reason of ["starting", "stopped", "not started", null, ""]) {
      assert.equal(
        pairingErrorMessage(
          true,
          false,
          { permitted: true, serving: false, reason },
          false,
        ),
        null,
        reason,
      );
    }
  });

  it("surfaces bind failures and poll unreachability on the laptop", () => {
    assert.equal(
      pairingErrorMessage(
        true,
        false,
        { permitted: true, serving: false, reason: "remote server failed to bind" },
        false,
      ),
      "remote server failed to bind",
    );
    assert.equal(pairingErrorMessage(true, false, null, true), "Remote is not available");
  });

  it("surfaces permit failures immediately", () => {
    assert.equal(
      pairingErrorMessage(
        true,
        false,
        { permitted: false, serving: false, reason: "LTX_AUTH_TOKEN is empty" },
        false,
      ),
      "LTX_AUTH_TOKEN is empty",
    );
  });
});
