import { useCallback, useEffect, useSyncExternalStore } from "react";
import { useNavigate } from "react-router";

import { historyShortcutAction } from "@/lib/history-shortcuts";
import { hasOpenModal } from "@/lib/open-modal";

type HistoryAvailability = {
  canGoBack: boolean;
  canGoForward: boolean;
};

type NavigationLike = HistoryAvailability & {
  addEventListener(type: "currententrychange", listener: () => void): void;
  removeEventListener(type: "currententrychange", listener: () => void): void;
};

const NONE: HistoryAvailability = { canGoBack: false, canGoForward: false };
let snapshot: HistoryAvailability = NONE;

function browserNavigation(): NavigationLike | null {
  const candidate: unknown = Reflect.get(window, "navigation");
  if (
    typeof candidate !== "object" ||
    candidate === null ||
    !("canGoBack" in candidate) ||
    !("canGoForward" in candidate)
  ) {
    return null;
  }
  return candidate as NavigationLike;
}

function subscribe(onChange: () => void): () => void {
  const navigation = browserNavigation();
  if (navigation == null) return () => undefined;
  navigation.addEventListener("currententrychange", onChange);
  return () => navigation.removeEventListener("currententrychange", onChange);
}

function getSnapshot(): HistoryAvailability {
  const navigation = browserNavigation();
  if (navigation == null) return NONE;
  if (
    snapshot.canGoBack === navigation.canGoBack &&
    snapshot.canGoForward === navigation.canGoForward
  ) {
    return snapshot;
  }
  snapshot = {
    canGoBack: navigation.canGoBack,
    canGoForward: navigation.canGoForward,
  };
  return snapshot;
}

/**
 * Back and forward follow the browser history stack via the Navigation API.
 * Electron 41 exposes it, including entries that survive a reload. The move
 * itself is still the router's delta (`navigate(-1)` / `navigate(1)`).
 */
export function useHistoryNav(): HistoryAvailability & {
  goBack: () => void;
  goForward: () => void;
} {
  const navigate = useNavigate();
  const availability = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  const goBack = useCallback(() => {
    if (!getSnapshot().canGoBack) return;
    navigate(-1);
  }, [navigate]);
  const goForward = useCallback(() => {
    if (!getSnapshot().canGoForward) return;
    navigate(1);
  }, [navigate]);
  return { ...availability, goBack, goForward };
}

export function useHistoryShortcuts(): void {
  const { goBack, goForward } = useHistoryNav();

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      const target = event.target;
      const inTextEntry =
        target instanceof Element &&
        target.closest('input, textarea, select, [contenteditable="true"]') != null;
      const { canGoBack, canGoForward } = getSnapshot();
      const action = historyShortcutAction({
        platform: document.documentElement.dataset.platform,
        code: event.code,
        altKey: event.altKey,
        metaKey: event.metaKey,
        ctrlKey: event.ctrlKey,
        shiftKey: event.shiftKey,
        inTextEntry,
        modalOpen: hasOpenModal(document),
        canGoBack,
        canGoForward,
      });
      if (action == null) return;
      event.preventDefault();
      event.stopPropagation();
      if (action === "back") goBack();
      else goForward();
    };
    window.addEventListener("keydown", onKeyDown, { capture: true });
    return () => window.removeEventListener("keydown", onKeyDown, { capture: true });
  }, [goBack, goForward]);
}
