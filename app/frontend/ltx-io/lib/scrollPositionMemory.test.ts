import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  createScrollPositionMemory,
  scrollStorageKey,
  type ScrollStorage,
} from "./scrollPositionMemory.ts";

function fakeStorage(initial: Record<string, string> = {}) {
  const data = new Map(Object.entries(initial));
  const storage: ScrollStorage = {
    getItem: (key) => data.get(key) ?? null,
    setItem: (key, value) => {
      data.set(key, value);
    },
  };
  return { storage, data };
}

describe("createScrollPositionMemory", () => {
  it("restores a saved position, and ignores a value that is not a number of 0 or more", () => {
    const key = scrollStorageKey("a");
    assert.equal(
      createScrollPositionMemory(fakeStorage({ [key]: "4210" }).storage).begin("a", true),
      4210,
    );
    // Each of these must open at the top. A bad value must not reach scrollTop.
    for (const bad of ["-5", "abc", "NaN", "Infinity", "0"]) {
      const memory = createScrollPositionMemory(fakeStorage({ [key]: bad }).storage);
      assert.equal(memory.begin("a", true), null, `stored "${bad}"`);
    }
  });

  it("does not save before the restore step, so it cannot overwrite the saved position with 0", () => {
    const key = scrollStorageKey("a");
    const { storage, data } = fakeStorage({ [key]: "500" });
    const memory = createScrollPositionMemory(storage);

    // The scroll event of a page that is still empty reports 0.
    memory.save("a", 0);
    assert.equal(data.get(key), "500");

    assert.equal(memory.begin("a", true), 500);
  });

  it("does not save under a page whose restore step has not run (page switch)", () => {
    const keyA = scrollStorageKey("a");
    const keyB = scrollStorageKey("b");
    const { storage, data } = fakeStorage({ [keyA]: "800", [keyB]: "300" });
    const memory = createScrollPositionMemory(storage);

    memory.begin("a", true);
    // The page changed to "b" and its content is gone, so the scroller reports 0.
    memory.save("b", 0);
    assert.equal(data.get(keyA), "800");
    assert.equal(data.get(keyB), "300");

    memory.begin("b", true);
    // A late event for the old page must not write either.
    memory.save("a", 0);
    assert.equal(data.get(keyA), "800");
  });

  it("with restore false, skips the saved position but still saves later scrolls", () => {
    const key = scrollStorageKey("a");
    const { storage, data } = fakeStorage({ [key]: "500" });
    const memory = createScrollPositionMemory(storage);

    // A link to one result sets the position, so the saved one must not be applied.
    assert.equal(memory.begin("a", false), null);
    assert.equal(memory.isTracking("a"), true);
    memory.save("a", 120.4);
    assert.equal(data.get(key), "120");
  });

  it("does not fail when the storage is blocked or throws", () => {
    const throwing: ScrollStorage = {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("quota");
      },
    };
    for (const storage of [throwing, null]) {
      const memory = createScrollPositionMemory(storage);
      assert.equal(memory.begin("a", true), null);
      assert.doesNotThrow(() => memory.save("a", 100));
    }
  });
});
