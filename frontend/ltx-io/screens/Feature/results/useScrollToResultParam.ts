import { useEffect, useRef, type RefObject } from "react";

import {
  getScrollParent,
  isResultMissing,
  scrollTopToCenter,
} from "./useResultsFeedScroll";

export function useScrollToResultParam(
  feedRootRef: RefObject<HTMLElement | null>,
  generations: readonly { id: string }[],
  resultId: string | null,
  listSettled: boolean,
  onSettled: (resultId: string) => void,
  onMissing: (resultId: string) => void,
  /** The feed is virtualized, so a row far from the viewport is not in the DOM. */
  scrollToIndex: (index: number) => void,
  /** Changes when the set of mounted rows changes. It re-runs the lookup after a jump. */
  renderedKey: string,
): void {
  const scrolledResultIdRef = useRef<string | null>(null);
  const missingResultIdRef = useRef<string | null>(null);
  const onSettledRef = useRef(onSettled);
  onSettledRef.current = onSettled;
  const onMissingRef = useRef(onMissing);
  onMissingRef.current = onMissing;
  const scrollToIndexRef = useRef(scrollToIndex);
  scrollToIndexRef.current = scrollToIndex;

  useEffect(() => {
    if (resultId == null) {
      scrolledResultIdRef.current = null;
      missingResultIdRef.current = null;
      return;
    }
    if (scrolledResultIdRef.current === resultId) return;
    const match = feedRootRef.current?.querySelector(
      `[data-generation-id="${CSS.escape(resultId)}"]`,
    );
    if (!(match instanceof HTMLElement)) {
      const index = generations.findIndex((row) => row.id === resultId);
      if (index >= 0) {
        // In the list but not mounted. The jump mounts the row and `renderedKey` re-runs this.
        scrollToIndexRef.current(index);
        return;
      }
      if (
        isResultMissing({ resultId, found: false, listSettled }) &&
        missingResultIdRef.current !== resultId
      ) {
        missingResultIdRef.current = resultId;
        onMissingRef.current(resultId);
      }
      return;
    }
    const scroller = getScrollParent(match);
    scrolledResultIdRef.current = resultId;
    // No smooth scroll. The virtualizer sets the scroll position while rows are measured,
    // and that cancels a smooth scroll.
    if (scroller == null) {
      match.scrollIntoView({ block: "center", behavior: "auto" });
      onSettledRef.current(resultId);
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
    scroller.scrollTo({ top, behavior: "auto" });
    onSettledRef.current(resultId);
  }, [feedRootRef, generations, listSettled, renderedKey, resultId]);
}
