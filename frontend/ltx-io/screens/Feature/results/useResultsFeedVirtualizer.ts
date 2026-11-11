import {
  defaultRangeExtractor,
  measureElement as measureRowElement,
  useVirtualizer,
  type Range,
} from "@tanstack/react-virtual";
import {
  useCallback,
  useLayoutEffect,
  useRef,
  useState,
  type RefObject,
} from "react";

import { createRowHeightEstimator } from "./rowHeightEstimator";
import { getScrollParent } from "./useResultsFeedScroll";

const OVERSCAN_ROWS = 3;

type FeedLayout = { width: number; gap: number; scrollMargin: number };

/**
 * The newest row stays mounted when it is far off screen. The scroll-up button and the
 * follow-newest pin read its position, and a mounted row keeps that position valid.
 */
function keepNewestMounted(range: Range): number[] {
  const indexes = defaultRangeExtractor(range);
  if (range.count === 0 || indexes[0] === 0) return indexes;
  return [0, ...indexes];
}

/** Distance from the top of the scroller content to the top of the list. */
function listOffsetInScroller(list: HTMLElement, scroller: HTMLElement): number {
  return (
    list.getBoundingClientRect().top -
    scroller.getBoundingClientRect().top +
    scroller.scrollTop
  );
}

/**
 * Virtualizes the results feed inside its scroll parent. Rows have different heights
 * (a failed row is shorter than a playing one), so each mounted row is measured.
 */
export function useResultsFeedVirtualizer({
  feedRootRef,
  generationIds,
}: {
  feedRootRef: RefObject<HTMLElement | null>;
  generationIds: readonly string[];
}) {
  const [listElement, setListElement] = useState<HTMLDivElement | null>(null);
  const [scrollElement, setScrollElement] = useState<HTMLElement | null>(null);
  const [layout, setLayout] = useState<FeedLayout>({
    width: 0,
    gap: 0,
    scrollMargin: 0,
  });

  useLayoutEffect(() => {
    const feedRoot = feedRootRef.current;
    if (!feedRoot || !listElement) return;

    let scroller: HTMLElement | null = null;
    let marginFrame = 0;

    // Width, gap, and the scroller change only when the window or the feed resizes.
    const syncLayout = () => {
      const nextScroller = getScrollParent(feedRoot);
      if (nextScroller !== scroller) {
        scroller?.removeEventListener("scroll", onScroll);
        scroller = nextScroller;
        scroller?.addEventListener("scroll", onScroll, { passive: true });
        setScrollElement(nextScroller);
      }
      if (!scroller) return;
      const next: FeedLayout = {
        width: listElement.clientWidth,
        // The CSS row gap of the feed. Reading it keeps one source for the spacing.
        gap: Number.parseFloat(getComputedStyle(feedRoot).rowGap) || 0,
        scrollMargin: listOffsetInScroller(listElement, scroller),
      };
      setLayout((prev) =>
        prev.width === next.width &&
        prev.gap === next.gap &&
        prev.scrollMargin === next.scrollMargin
          ? prev
          : next,
      );
    };

    // In the stacked layout the content above the list can change height without a resize
    // of the feed. The list then moves in the scroller, so read its offset again while the
    // user scrolls. Read once for each frame, and only the offset.
    const syncScrollMargin = () => {
      marginFrame = 0;
      if (!scroller) return;
      const scrollMargin = listOffsetInScroller(listElement, scroller);
      setLayout((prev) =>
        prev.scrollMargin === scrollMargin ? prev : { ...prev, scrollMargin },
      );
    };
    function onScroll() {
      if (marginFrame === 0) marginFrame = requestAnimationFrame(syncScrollMargin);
    }

    syncLayout();
    const resize = new ResizeObserver(syncLayout);
    resize.observe(feedRoot);
    window.addEventListener("resize", syncLayout);
    return () => {
      scroller?.removeEventListener("scroll", onScroll);
      if (marginFrame !== 0) cancelAnimationFrame(marginFrame);
      resize.disconnect();
      window.removeEventListener("resize", syncLayout);
    };
  }, [feedRootRef, listElement]);

  const estimatorRef = useRef<ReturnType<typeof createRowHeightEstimator>>();
  estimatorRef.current ??= createRowHeightEstimator();
  const estimator = estimatorRef.current;
  const widthRef = useRef(layout.width);
  widthRef.current = layout.width;

  // A new function on each render makes the library rebuild all row measurements on each
  // render. Keep it stable between lists.
  const getItemKey = useCallback(
    (index: number) => generationIds[index] ?? index,
    [generationIds],
  );

  const virtualizer = useVirtualizer({
    count: generationIds.length,
    getScrollElement: () => scrollElement,
    estimateSize: () => estimator.estimate(widthRef.current),
    measureElement: (element, entry, instance) => {
      const height = measureRowElement(element, entry, instance);
      const key = instance.options.getItemKey(instance.indexFromElement(element));
      estimator.record(key, height, widthRef.current);
      return height;
    },
    getItemKey,
    overscan: OVERSCAN_ROWS,
    gap: layout.gap,
    scrollMargin: layout.scrollMargin,
    rangeExtractor: keepNewestMounted,
  });

  // A new width changes every row height, because the video frame scales with the width.
  // A row that is off screen keeps its old cached height, so its offset would be wrong.
  // Drop the cache, then measure the mounted rows again. They do not report on their own,
  // because the layout change happened before the cache was cleared.
  const feedWidth = layout.width;
  useLayoutEffect(() => {
    if (!listElement || feedWidth === 0) return;
    virtualizer.measure();
    for (const row of listElement.querySelectorAll<HTMLElement>("[data-index]")) {
      virtualizer.measureElement(row);
    }
  }, [virtualizer, listElement, feedWidth]);

  const virtualItems = virtualizer.getVirtualItems();
  const lastRenderedIndex = virtualItems[virtualItems.length - 1]?.index ?? -1;

  const scrollToIndex = useCallback(
    (index: number) => {
      virtualizer.scrollToIndex(index, { align: "center", behavior: "auto" });
    },
    [virtualizer],
  );

  return {
    setListElement,
    scrollElement,
    virtualItems,
    totalSize: virtualizer.getTotalSize(),
    scrollMargin: layout.scrollMargin,
    measureElement: virtualizer.measureElement,
    scrollToIndex,
    /** Changes when the set of mounted rows changes. */
    renderedKey: `${virtualItems.length}:${lastRenderedIndex}`,
  };
}
