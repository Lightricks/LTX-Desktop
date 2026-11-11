import { useSyncExternalStore } from "react";

export const SKIP_DELETE_CONFIRMATION_STORAGE_KEY = "ltxio.skipDeleteConfirmation";

type Listener = () => void;

const listeners = new Set<Listener>();
let memoryFallbackSkip = false;
let storageListenerAttached = false;

function notifyListeners(): void {
  for (const listener of listeners) {
    listener();
  }
}

function ensureStorageListener(): void {
  if (
    storageListenerAttached ||
    typeof window === "undefined" ||
    typeof window.addEventListener !== "function"
  ) {
    return;
  }
  storageListenerAttached = true;
  window.addEventListener("storage", (event) => {
    if (event.key !== null && event.key !== SKIP_DELETE_CONFIRMATION_STORAGE_KEY) {
      return;
    }
    notifyListeners();
  });
}

export function getSkipDeleteConfirmation(): boolean {
  try {
    const storage = globalThis.localStorage;
    if (!storage) return memoryFallbackSkip;
    return storage.getItem(SKIP_DELETE_CONFIRMATION_STORAGE_KEY) === "true";
  } catch {
    return memoryFallbackSkip;
  }
}

export function setSkipDeleteConfirmation(skip: boolean): void {
  const next = Boolean(skip);
  const previous = getSkipDeleteConfirmation();
  memoryFallbackSkip = next;
  try {
    if (next) {
      globalThis.localStorage?.setItem(SKIP_DELETE_CONFIRMATION_STORAGE_KEY, "true");
    } else {
      globalThis.localStorage?.removeItem(SKIP_DELETE_CONFIRMATION_STORAGE_KEY);
    }
  } catch {
    // localStorage can throw in private mode or restricted environments.
  }
  if (getSkipDeleteConfirmation() !== previous) {
    notifyListeners();
  }
}

export function subscribeSkipDeleteConfirmation(listener: Listener): () => void {
  ensureStorageListener();
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function useDeleteConfirmationPreference(): {
  skipDeleteConfirmation: boolean;
  setSkipDeleteConfirmation: (skip: boolean) => void;
} {
  const skipDeleteConfirmation = useSyncExternalStore(
    subscribeSkipDeleteConfirmation,
    getSkipDeleteConfirmation,
    () => false,
  );
  return {
    skipDeleteConfirmation,
    setSkipDeleteConfirmation,
  };
}
