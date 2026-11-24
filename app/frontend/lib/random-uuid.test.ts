import assert from "node:assert/strict";
import { it } from "node:test";

import { randomUUID } from "./random-uuid.ts";

it("still creates v4 ids over plain-http LAN, where crypto.randomUUID is missing", () => {
  const original = Object.getOwnPropertyDescriptor(crypto, "randomUUID");
  Object.defineProperty(crypto, "randomUUID", { value: undefined, configurable: true });
  try {
    const ids = [randomUUID(), randomUUID()];
    for (const id of ids) {
      assert.match(id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    }
    assert.notEqual(ids[0], ids[1]);
  } finally {
    if (original) Object.defineProperty(crypto, "randomUUID", original);
    else delete (crypto as { randomUUID?: unknown }).randomUUID;
  }
});
