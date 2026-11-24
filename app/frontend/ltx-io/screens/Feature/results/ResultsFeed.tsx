import { clsx } from "clsx";
import { useCallback, useMemo, useState, type ComponentType } from "react";
import { useSearchParams } from "react-router";

import { ActivityCircular } from "@ds/ActivityCircular/ActivityCircular";
import { Button } from "@ds/Button/Button";
import ArrowUpIcon from "@ds/assets/Icons/Arrow/Up/Small.svg?react";

import {
  progressPercentByGenerationId,
  remainingSecondsByGenerationId,
} from "../../../components/GenerationQueue/generationQueuePanelModel";
import {
  createToast,
  DEFAULT_AUTOCLOSE_MS,
} from "../../../components/shared/Toast/toastService";
import { useGenerationQueue } from "../../../hooks/useGenerationQueue";
import { useScrollPositionMemory } from "../../../hooks/useScrollPositionMemory";
import { isInFlightGeneration, type Generation } from "../../../lib/resultsFeedModel";
import { useExploreRuntime } from "../../../runtime/ExploreRuntime";
import type { ResultFrameProps } from "./ResultFrame";
import { ResultStatus } from "./ResultStatus";
import { RESULT_NOT_FOUND_MESSAGE } from "./resultActionsCopy";
import { isStillFocusingResult, useResultsFeedScroll } from "./useResultsFeedScroll";
import { useResultsFeedVirtualizer } from "./useResultsFeedVirtualizer";
import { useScrollToResultParam } from "./useScrollToResultParam";
import styles from "./ResultsFeed.module.scss";

export type EmptyStateAdapter = ComponentType;

function ScrollUpButton({
  scrollToNewest,
  showScrollUp,
  pulse,
  clearPulse,
}: {
  scrollToNewest: () => void;
  showScrollUp: boolean;
  pulse: boolean;
  clearPulse: () => void;
}) {
  return (
    <div
      className={clsx(
        styles.scrollUp,
        showScrollUp && styles.scrollUpVisible,
        pulse && styles.scrollUpPulse,
      )}
      onAnimationEnd={clearPulse}
    >
      <div className={styles.scrollUpInner}>
        <Button
          appearance="overlay"
          hierarchy="secondary"
          size="xl"
          leftIcon={<ArrowUpIcon />}
          label="Scroll Up"
          aria-label="Scroll to newest result"
          onClick={scrollToNewest}
        />
      </div>
    </div>
  );
}

export function ResultsFeed({
  pageId,
  generations,
  isLoading,
  isFetching,
  loadError,
  onCancel,
  onRetry,
  onDelete,
  ResultFrameAdapter,
  EmptyStateAdapter,
}: {
  /** Names the page for the remembered scroll position. */
  pageId: string;
  generations: Generation[];
  isLoading: boolean;
  isFetching: boolean;
  loadError: string | null;
  onCancel: (id: string) => void;
  onRetry: (id: string) => void;
  onDelete: (id: string) => void;
  ResultFrameAdapter: ComponentType<ResultFrameProps>;
  EmptyStateAdapter?: EmptyStateAdapter;
}) {
  const [searchParams, setSearchParams] = useSearchParams();
  const resultId = searchParams.get("result");
  const [settledResultId, setSettledResultId] = useState<string | null>(null);
  const { api, generationPolling } = useExploreRuntime();
  const isProcessing = generations.some(isInFlightGeneration);
  const queueQuery = useGenerationQueue({
    api,
    generationPolling,
    // Progress % only needs the slow background rate; the queue panel speeds it up when opened.
    isLive: false,
  });
  const progressById = useMemo(
    () => progressPercentByGenerationId(queueQuery.data),
    [queueQuery.data],
  );
  const remainingById = useMemo(
    () => remainingSecondsByGenerationId(queueQuery.data),
    [queueQuery.data],
  );
  const {
    feedRootRef,
    newestRef,
    scrollToNewest,
    unpinFromNewest,
    showScrollUp,
    pulse,
    clearPulse,
  } = useResultsFeedScroll({
    itemCount: generations.length,
    isProcessing,
    focusingResult: isStillFocusingResult(resultId, settledResultId),
  });

  const generationIds = useMemo(
    () => generations.map((generation) => generation.id),
    [generations],
  );
  const {
    setListElement,
    scrollElement,
    virtualItems,
    totalSize,
    scrollMargin,
    measureElement,
    scrollToIndex,
    renderedKey,
  } = useResultsFeedVirtualizer({ feedRootRef, generationIds });

  // A restored position must not be undone by follow-newest, which would scroll to the top.
  // A ?result= link sets its own position, so it skips the restore.
  useScrollPositionMemory({
    pageId,
    scrollElement,
    ready: generations.length > 0,
    restore: resultId == null,
    onRestored: unpinFromNewest,
  });

  const newestId = generations[0]?.id ?? null;
  const onResultScrolled = useCallback(
    (id: string) => {
      if (id !== newestId) unpinFromNewest();
      setSettledResultId(id);
    },
    [newestId, unpinFromNewest],
  );
  // A Queue link can point at a generation this page does not list, for example one
  // that was deleted. Tell the user, then drop the stale param.
  const onResultMissing = useCallback(
    (id: string) => {
      setSettledResultId(id);
      createToast({
        message: RESULT_NOT_FOUND_MESSAGE,
        toastType: "warning",
        closeable: true,
        autoCloseAfter: DEFAULT_AUTOCLOSE_MS,
      });
      setSearchParams(
        (current) => {
          const next = new URLSearchParams(current);
          next.delete("result");
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );
  useScrollToResultParam(
    feedRootRef,
    generations,
    resultId,
    !isLoading && !isFetching && !loadError,
    onResultScrolled,
    onResultMissing,
    scrollToIndex,
    renderedKey,
  );

  const isEmpty = generations.length === 0 && !isLoading && !loadError;
  const isLoadError = Boolean(loadError) && generations.length === 0 && !isLoading;

  return (
    <div
      ref={feedRootRef}
      className={clsx(styles.outputStack, (isEmpty || isLoadError) && styles.outputStackCentered)}
    >
      <ScrollUpButton
        scrollToNewest={scrollToNewest}
        showScrollUp={showScrollUp}
        pulse={pulse}
        clearPulse={clearPulse}
      />
      {isEmpty && EmptyStateAdapter ? (
        <div ref={newestRef} className={styles.exampleEmpty}>
          <EmptyStateAdapter />
        </div>
      ) : null}
      {isLoadError ? (
        <div ref={newestRef} className={styles.exampleEmpty}>
          <ResultStatus
            title="Couldn't load results"
            body={loadError ?? undefined}
          />
        </div>
      ) : null}
      {isLoading && generations.length === 0 ? (
        <div
          ref={newestRef}
          className={styles.historyLoading}
          aria-busy="true"
          aria-label="Loading results"
        >
          <ActivityCircular appearance="default" size={16} />
        </div>
      ) : null}
      {generations.length > 0 ? (
        <div
          ref={setListElement}
          className={styles.virtualList}
          style={{ height: totalSize }}
        >
          {virtualItems.map((virtualItem) => {
            const generation = generations[virtualItem.index];
            if (!generation) return null;
            return (
              <div
                key={virtualItem.key}
                data-index={virtualItem.index}
                ref={measureElement}
                className={styles.virtualRow}
                style={{
                  transform: `translateY(${virtualItem.start - scrollMargin}px)`,
                }}
              >
                <div
                  ref={generation.id === newestId ? newestRef : undefined}
                  data-generation-id={generation.id}
                  className={
                    isInFlightGeneration(generation)
                      ? styles.resultGroup
                      : undefined
                  }
                >
                  <ResultFrameAdapter
                    generation={generation}
                    onCancel={onCancel}
                    onRetry={onRetry}
                    onDelete={onDelete}
                    progressPercent={progressById.get(generation.id)}
                    remainingSeconds={remainingById.get(generation.id)}
                  />
                </div>
              </div>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
