import assert from "node:assert/strict";
import { test } from "node:test";

import {
  SKIP_DELETE_CONFIRMATION_STORAGE_KEY,
  getSkipDeleteConfirmation,
  setSkipDeleteConfirmation,
  subscribeSkipDeleteConfirmation,
} from "./delete-confirmation-preference.ts";

function installMemoryStorage(backing = new Map<string, string>()): {
  storage: Storage;
  restore: () => void;
} {
  const originalDescriptor = Object.getOwnPropertyDescriptor(
    globalThis,
    "localStorage",
  );
  const storage: Storage = {
    get length() {
      return backing.size;
    },
    clear() {
      backing.clear();
    },
    getItem(key: string) {
      return backing.get(key) ?? null;
    },
    key(index: number) {
      return Array.from(backing.keys())[index] ?? null;
    },
    removeItem(key: string) {
      backing.delete(key);
    },
    setItem(key: string, value: string) {
      backing.set(key, String(value));
    },
  };
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    writable: true,
    value: storage,
  });
  return {
    storage,
    restore() {
      if (originalDescriptor) {
        Object.defineProperty(globalThis, "localStorage", originalDescriptor);
      } else {
        Reflect.deleteProperty(globalThis, "localStorage");
      }
    },
  };
}

test("deleteConfirmationPreference_persistsAndResetsSkipFlagWhenToggled", () => {
  const { storage, restore } = installMemoryStorage();
  const notifications: boolean[] = [];
  const unsubscribe = subscribeSkipDeleteConfirmation(() => {
    notifications.push(getSkipDeleteConfirmation());
  });

  try {
    assert.equal(getSkipDeleteConfirmation(), false);

    setSkipDeleteConfirmation(true);
    assert.equal(getSkipDeleteConfirmation(), true);
    assert.equal(storage.getItem(SKIP_DELETE_CONFIRMATION_STORAGE_KEY), "true");
    assert.deepEqual(notifications, [true]);

    setSkipDeleteConfirmation(false);
    assert.equal(getSkipDeleteConfirmation(), false);
    assert.equal(storage.getItem(SKIP_DELETE_CONFIRMATION_STORAGE_KEY), null);
    assert.deepEqual(notifications, [true, false]);
  } finally {
    setSkipDeleteConfirmation(false);
    unsubscribe();
    restore();
  }
});
