import { useEffect, useLayoutEffect, useRef } from "react";

import {
  createScrollPositionMemory,
  getSessionScrollStorage,
} from "../lib/scrollPositionMemory";

/**
 * Remembers the scroll position (Y) of a page for the browser session, and restores it
 * when the page opens again. The session is per window: Desktop and the remote app do not
 * share it. The save and restore rules are in `scrollPositionMemory.ts`.
 */
export function useScrollPositionMemory({
  pageId,
  scrollElement,
  ready,
  restore = true,
  onRestored,
}: {
  /** One id for each page, for example a feature id. */
  pageId: string;
  /** The element that scrolls. */
  scrollElement: HTMLElement | null;
  /** The content is in the DOM, so the saved position can be reached. */
  ready: boolean;
  /** False when something else sets the position, for example a link to one result. */
  restore?: boolean;
  /** Runs after a saved position was applied. */
  onRestored?: () => void;
}): void {
  const memoryRef = useRef<ReturnType<typeof createScrollPositionMemory>>();
  memoryRef.current ??= createScrollPositionMemory(getSessionScrollStorage());
  const memory = memoryRef.current;
  const pageIdRef = useRef(pageId);
  pageIdRef.current = pageId;
  const onRestoredRef = useRef(onRestored);
  onRestoredRef.current = onRestored;

  useLayoutEffect(() => {
    if (!scrollElement || !ready || memory.isTracking(pageId)) return;
    const saved = memory.begin(pageId, restore);
    if (saved != null) {
      scrollElement.scrollTop = saved;
      onRestoredRef.current?.();
    }
  }, [memory, pageId, scrollElement, ready, restore]);

  useEffect(() => {
    if (!scrollElement) return;
    let frame = 0;
    const save = () => {
      frame = 0;
      memory.save(pageIdRef.current, scrollElement.scrollTop);
    };
    const onScroll = () => {
      if (frame === 0) frame = requestAnimationFrame(save);
    };
    scrollElement.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      scrollElement.removeEventListener("scroll", onScroll);
      // A detached element reports scrollTop 0. Do not save after cleanup.
      if (frame !== 0) cancelAnimationFrame(frame);
    };
  }, [memory, scrollElement]);
}
