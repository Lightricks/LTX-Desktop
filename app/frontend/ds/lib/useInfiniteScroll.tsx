// Infinite-scroll hook for the design system. Keeps `frontend/ds/` self-contained.
import { useCallback, useEffect, useRef, useState } from "react";
import { useInView } from "react-intersection-observer";

import { ActivityCircular } from "@ds/ActivityCircular/ActivityCircular";

import styles from "./useInfiniteScroll.module.scss";

export type ContentState = "loading" | "error" | "empty" | "content";

export function getContentState({
  isLoading,
  error,
  isEmpty,
}: {
  isLoading: boolean;
  error: unknown;
  isEmpty: boolean;
}): ContentState {
  if (error && isEmpty) return "error";
  if (isLoading && isEmpty) return "loading";
  if (isEmpty) return "empty";
  return "content";
}

export interface UseInfiniteScrollOptions {
  /** The scrollable container element. Use `useRootScrollContainer()` for pages inside Root. */
  scrollContainer?: HTMLElement | null;
  /** Whether there is a next page to load */
  hasNextPage?: boolean;
  /** Callback function to load the next page */
  onNextPage?: () => void;
  /** Distance from bottom in pixels to trigger loading more items */
  bottomThreshold?: number;
  /** Total number of items currently loaded */
  itemsCount: number;
  /** Whether the query is fetching the next page */
  isFetchingNextPage?: boolean;
  /** Whether to fill container when it doesn't have enough content */
  fillContainer?: boolean;
}

export interface UseInfiniteScrollResult {
  /** Ref to attach to the scrollable container (used when scrollContainer is not provided) */
  scrollRef: React.RefObject<HTMLDivElement>;
  /** Scrollable element resolved from scrollRef. */
  scrollElement: HTMLElement | null;
  /** Scrolls the scrollable element back to the top. */
  scrollToTop: (behavior?: ScrollBehavior) => void;
  /** Sentinel component to render at the end of the content */
  InfiniteScrollSentinel: React.FC;
  /** Loading spinner rendered while the next page is fetching */
  InfiniteScrollLoadingMore: React.FC;
}

interface SentinelProps {
  sentinelRef: (node?: Element | null) => void;
}

function InfiniteScrollSentinelComponent({ sentinelRef }: SentinelProps) {
  return (
    <div ref={sentinelRef} style={{ width: "100%", height: 0, visibility: "hidden" }} />
  );
}

/**
 * Infinite scroll hook using react-intersection-observer.
 * Attach scrollRef to a scrollable container and render InfiniteScrollSentinel at the end.
 */
export function useInfiniteScroll({
  hasNextPage = false,
  onNextPage,
  bottomThreshold = 100,
  itemsCount,
  isFetchingNextPage = false,
  fillContainer = false,
  scrollContainer,
}: UseInfiniteScrollOptions): UseInfiniteScrollResult {
  const scrollRef = useRef<HTMLDivElement>(null);
  const scrollElement = scrollContainer ?? scrollRef.current;

  const prevItemsCountRef = useRef<number>(itemsCount);
  const fillCheckTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Synchronous loading guard to prevent race conditions.
  // isFetchingNextPage from React Query updates asynchronously (next render cycle),
  // so without this guard, multiple onNextPage calls could fire before the query state updates.
  const [isLoadingNext, setIsLoadingNext] = useState(false);

  const { ref: sentinelRef, inView } = useInView({
    threshold: 0,
    rootMargin: `0px 0px ${bottomThreshold}px 0px`,
    root: scrollElement,
    skip: !scrollElement,
  });

  const triggerNextPage = useCallback(() => {
    if (isFetchingNextPage || isLoadingNext || !hasNextPage || !onNextPage) {
      return false;
    }
    setIsLoadingNext(true);
    onNextPage();
    return true;
  }, [onNextPage, isFetchingNextPage, isLoadingNext, hasNextPage]);

  const checkAndFillContainer = useCallback(() => {
    if (isFetchingNextPage || isLoadingNext || !hasNextPage || !fillContainer) {
      return false;
    }

    if (!scrollElement) return false;

    const { scrollHeight, clientHeight } = scrollElement;

    if (scrollHeight <= clientHeight && hasNextPage) {
      return triggerNextPage();
    }
    return false;
  }, [
    triggerNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoadingNext,
    fillContainer,
    scrollElement,
  ]);

  const startContentChecks = useCallback(() => {
    if (fillCheckTimeoutRef.current) {
      clearTimeout(fillCheckTimeoutRef.current);
    }

    fillCheckTimeoutRef.current = setTimeout(() => {
      checkAndFillContainer();
      fillCheckTimeoutRef.current = null;
    }, 100);
  }, [checkAndFillContainer]);

  useEffect(() => {
    if (!isFetchingNextPage) {
      setIsLoadingNext(false);
    }
  }, [isFetchingNextPage]);

  useEffect(() => {
    if (itemsCount !== prevItemsCountRef.current) {
      prevItemsCountRef.current = itemsCount;

      if (fillContainer && hasNextPage && !isFetchingNextPage) {
        startContentChecks();
      }
    }
  }, [itemsCount, hasNextPage, isFetchingNextPage, fillContainer, startContentChecks]);

  useEffect(() => {
    if (inView && !isFetchingNextPage && !isLoadingNext && hasNextPage) {
      triggerNextPage();
    }
  }, [inView, triggerNextPage, isFetchingNextPage, isLoadingNext, hasNextPage]);

  useEffect(() => {
    if (fillContainer) {
      startContentChecks();
    }

    return () => {
      if (fillCheckTimeoutRef.current) {
        clearTimeout(fillCheckTimeoutRef.current);
      }
    };
  }, [fillContainer, startContentChecks]);

  const InfiniteScrollSentinel = useCallback(
    function InfiniteScrollSentinel() {
      return <InfiniteScrollSentinelComponent sentinelRef={sentinelRef} />;
    },
    [sentinelRef],
  );

  const InfiniteScrollLoadingMore = useCallback(
    function InfiniteScrollLoadingMore() {
      if (!isFetchingNextPage) return null;
      return (
        <div className={styles.loadingMore}>
          <ActivityCircular size={32} />
        </div>
      );
    },
    [isFetchingNextPage],
  );

  const scrollToTop = useCallback(
    (behavior: ScrollBehavior = "smooth") => {
      scrollElement?.scrollTo({ top: 0, behavior });
    },
    [scrollElement],
  );

  return {
    scrollRef,
    scrollElement,
    scrollToTop,
    InfiniteScrollSentinel,
    InfiniteScrollLoadingMore,
  };
}
