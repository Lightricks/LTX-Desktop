import { useEffect, useRef, type RefObject } from "react";

import { getScrollParent, scrollTopToCenter } from "./useResultsFeedScroll";

export function useScrollToResultParam(
  feedRootRef: RefObject<HTMLElement | null>,
  generations: readonly unknown[],
  resultId: string | null,
  reduceMotion: boolean | null,
  onSettled: (resultId: string) => void,
): void {
  const scrolledResultIdRef = useRef<string | null>(null);
  const onSettledRef = useRef(onSettled);
  onSettledRef.current = onSettled;

  useEffect(() => {
    if (resultId == null) {
      scrolledResultIdRef.current = null;
      return;
    }
    if (scrolledResultIdRef.current === resultId) return;
    const match = feedRootRef.current?.querySelector(
      `[data-generation-id="${CSS.escape(resultId)}"]`,
    );
    if (!(match instanceof HTMLElement)) return;
    const scroller = getScrollParent(match);
    scrolledResultIdRef.current = resultId;
    const settle = () => onSettledRef.current(resultId);
    if (scroller == null) {
      match.scrollIntoView({
        block: "center",
        behavior: reduceMotion ? "auto" : "smooth",
      });
      settle();
      return;
    }
    const targetRect = match.getBoundingClientRect();
    const scrollerRect = scroller.getBoundingClientRect();
    const top = scrollTopToCenter({
      scrollTop: scroller.scrollTop,
      scrollerTop: scrollerRect.top,
      scrollerHeight: scroller.clientHeight,
      scrollHeight: scroller.scrollHeight,
      targetTop: targetRect.top,
      targetHeight: targetRect.height,
    });
    const behavior = reduceMotion ? "auto" : "smooth";
    const alreadyThere = Math.abs(scroller.scrollTop - top) < 1;
    scroller.scrollTo({ top, behavior });
    // Keep follow-newest suppressed until the smooth scroll finishes, or it
    // cancels this scroll while the pin still thinks we are at the top.
    if (behavior === "auto" || alreadyThere) {
      settle();
      return;
    }
    scroller.addEventListener("scrollend", settle, { once: true });
    return () => scroller.removeEventListener("scrollend", settle);
  }, [feedRootRef, generations, reduceMotion, resultId]);
}
