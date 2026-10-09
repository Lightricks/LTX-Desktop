import { useCallback, useEffect, useRef, useState } from "react";

const SHOW_BUTTON_DELAY_MS = 350;
const SHOW_AWAY_PX = 24;
const HIDE_AWAY_PX = 8;

export function getScrollParent(el: HTMLElement | null): HTMLElement | null {
  let node = el?.parentElement ?? null;
  while (node) {
    const overflowY = getComputedStyle(node).overflowY;
    if (overflowY === "auto" || overflowY === "scroll") return node;
    node = node.parentElement;
  }
  return null;
}

/**
 * The newest card is tall. "Away" means its top has left the scroller, so a
 * short scroll shows the button. Hysteresis keeps a one-pixel bounce from
 * cancelling the show delay.
 */
export function isAwayFromNewestResult(
  currentlyAway: boolean,
  newestTop: number,
  scrollerTop: number,
): boolean {
  const offset = scrollerTop - newestTop;
  return offset > (currentlyAway ? HIDE_AWAY_PX : SHOW_AWAY_PX);
}

/**
 * ?result= blocks follow-newest only until that card has been scrolled into view.
 * After that, the pin follows the scroller again.
 */
export function isStillFocusingResult(
  resultId: string | null,
  settledResultId: string | null,
): boolean {
  return resultId != null && resultId !== settledResultId;
}

/** A Done click keeps ?result= set. Chasing the newest card would undo that scroll. */
export function shouldFollowNewestResult(input: {
  generationStarted: boolean;
  pinnedToNewest: boolean;
  focusingResult: boolean;
}): boolean {
  if (input.generationStarted) return true;
  if (input.focusingResult) return false;
  return input.pinnedToNewest;
}

/** Center a card in the feed scroller. `scrollIntoView` misses that scroller. */
export function scrollTopToCenter(input: {
  scrollTop: number;
  scrollerTop: number;
  scrollerHeight: number;
  scrollHeight: number;
  targetTop: number;
  targetHeight: number;
}): number {
  const next =
    input.scrollTop +
    input.targetTop -
    input.scrollerTop -
    (input.scrollerHeight - input.targetHeight) / 2;
  const maxScroll = Math.max(0, input.scrollHeight - input.scrollerHeight);
  return Math.min(Math.max(0, next), maxScroll);
}

function scrollTopForNewest(scroller: HTMLElement, newest: HTMLElement): number {
  const paddingTop = Number.parseFloat(getComputedStyle(scroller).paddingTop) || 0;
  const top =
    scroller.scrollTop +
    newest.getBoundingClientRect().top -
    scroller.getBoundingClientRect().top -
    paddingTop;
  return Math.max(0, top);
}

type UseResultsFeedScrollArgs = {
  itemCount: number;
  isProcessing: boolean;
  focusingResult: boolean;
};

export function useResultsFeedScroll({
  itemCount,
  isProcessing,
  focusingResult,
}: UseResultsFeedScrollArgs) {
  const feedRootRef = useRef<HTMLDivElement>(null);
  const pinnedToNewestRef = useRef(true);
  const wasProcessingRef = useRef(isProcessing);
  const prevItemCountRef = useRef(itemCount);
  const newestElRef = useRef<HTMLDivElement | null>(null);
  const awayRef = useRef(false);

  const [newestNode, setNewestNode] = useState<HTMLDivElement | null>(null);
  const [isAwayFromNewest, setIsAwayFromNewest] = useState(false);
  const [showScrollUp, setShowScrollUp] = useState(false);
  const [pulse, setPulse] = useState(false);

  const newestRef = useCallback((node: HTMLDivElement | null) => {
    newestElRef.current = node;
    setNewestNode((current) => (current === node ? current : node));
  }, []);

  useEffect(() => {
    const newest = newestNode;
    if (!newest) return;

    let scroller: HTMLElement | null = null;

    const publish = () => {
      if (!scroller) return;
      const away = isAwayFromNewestResult(
        awayRef.current,
        newest.getBoundingClientRect().top,
        scroller.getBoundingClientRect().top,
      );
      if (away === awayRef.current) return;
      awayRef.current = away;
      pinnedToNewestRef.current = !away;
      setIsAwayFromNewest(away);
    };

    const bindScroller = () => {
      const nextScroller = getScrollParent(newest);
      if (nextScroller === scroller) return;
      scroller?.removeEventListener("scroll", publish);
      scroller = nextScroller;
      scroller?.addEventListener("scroll", publish, { passive: true });
    };

    const onResize = () => {
      bindScroller();
      publish();
    };

    onResize();
    const resize = new ResizeObserver(onResize);
    resize.observe(newest);
    window.addEventListener("resize", onResize);

    return () => {
      scroller?.removeEventListener("scroll", publish);
      resize.disconnect();
      window.removeEventListener("resize", onResize);
    };
  }, [newestNode]);

  const unpinFromNewest = useCallback(() => {
    pinnedToNewestRef.current = false;
  }, []);

  const scrollToNewest = useCallback(() => {
    const newest = newestElRef.current;
    const scroller = newest ? getScrollParent(newest) : null;
    if (!newest || !scroller) return;
    scroller.scrollTo({
      top: scrollTopForNewest(scroller, newest),
      behavior: "smooth",
    });
  }, []);

  useEffect(() => {
    const generationStarted = isProcessing && !wasProcessingRef.current;
    wasProcessingRef.current = isProcessing;
    if (
      shouldFollowNewestResult({
        generationStarted,
        pinnedToNewest: pinnedToNewestRef.current,
        focusingResult,
      })
    ) {
      scrollToNewest();
    }
  }, [focusingResult, itemCount, isProcessing, scrollToNewest]);

  useEffect(() => {
    if (!isAwayFromNewest) {
      setShowScrollUp(false);
      return;
    }
    const timer = setTimeout(() => setShowScrollUp(true), SHOW_BUTTON_DELAY_MS);
    return () => clearTimeout(timer);
  }, [isAwayFromNewest]);

  useEffect(() => {
    if (showScrollUp && itemCount !== prevItemCountRef.current) {
      setPulse(false);
      requestAnimationFrame(() => setPulse(true));
    }
    prevItemCountRef.current = itemCount;
  }, [itemCount, showScrollUp]);

  const clearPulse = useCallback(() => {
    setPulse(false);
  }, []);

  return {
    feedRootRef,
    newestRef,
    scrollToNewest,
    unpinFromNewest,
    showScrollUp,
    pulse,
    clearPulse,
  };
}
