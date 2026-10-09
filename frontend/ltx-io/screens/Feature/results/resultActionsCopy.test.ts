import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { REMOVE_FROM_HISTORY_LABEL } from "./resultActionsCopy.ts";

describe("results delete copy", () => {
  it("labels generation hide as remove from history", () => {
    assert.equal(REMOVE_FROM_HISTORY_LABEL, "Remove from history");
  });
});
