import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { hasOpenModal } from "./open-modal.ts";

describe("hasOpenModal", () => {
  function root(present: { ariaModal: boolean; dialog: boolean }) {
    return {
      querySelector(selector: string) {
        const wantsAriaModal = selector.includes('[aria-modal="true"]');
        const wantsDialog = selector.includes('[role="dialog"]');
        if (present.ariaModal && wantsAriaModal) return { kind: "aria-modal" };
        if (present.dialog && wantsDialog) return { kind: "dialog" };
        return null;
      },
    };
  }

  it("treats a role=dialog overlay as open when it does not set aria-modal", () => {
    assert.equal(hasOpenModal(root({ ariaModal: false, dialog: true })), true);
    assert.equal(hasOpenModal(root({ ariaModal: true, dialog: false })), true);
    assert.equal(hasOpenModal(root({ ariaModal: false, dialog: false })), false);
  });
});
