import { clsx } from "clsx";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useCallback, useMemo, useState, type ComponentType } from "react";
import { useSearchParams } from "react-router";

import { ActivityCircular } from "@ds/ActivityCircular/ActivityCircular";
import { Button } from "@ds/Button/Button";
import ArrowUpIcon from "@ds/assets/Icons/Arrow/Up/Small.svg?react";

import { progressPercentByGenerationId } from "../../../components/GenerationQueue/generationQueuePanelModel";
import { useGenerationQueue } from "../../../hooks/useGenerationQueue";
import { isInFlightGeneration, type Generation } from "../../../lib/resultsFeedModel";
import { useExploreRuntime } from "../../../runtime/ExploreRuntime";
import type { ResultFrameProps } from "./ResultFrame";
import { ResultStatus } from "./ResultStatus";
import { isStillFocusingResult, useResultsFeedScroll } from "./useResultsFeedScroll";
import { useScrollToResultParam } from "./useScrollToResultParam";
import styles from "./ResultsFeed.module.scss";

const PUSH_S = 0.35;
const REVEAL_DELAY_S = 0.35;
const MOTION_EASE = [0, 0, 0.3, 1] as const;

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
  generations,
  isLoading,
  loadError,
  onCancel,
  onRetry,
  onDelete,
  ResultFrameAdapter,
  EmptyStateAdapter,
}: {
  generations: Generation[];
  isLoading: boolean;
  loadError: string | null;
  onCancel: (id: string) => void;
  onRetry: (id: string) => void;
  onDelete: (id: string) => void;
  ResultFrameAdapter: ComponentType<ResultFrameProps>;
  EmptyStateAdapter?: EmptyStateAdapter;
}) {
  const reduceMotion = useReducedMotion();
  const [searchParams] = useSearchParams();
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

  const newestId = generations[0]?.id ?? null;
  const onResultScrolled = useCallback(
    (id: string) => {
      if (id !== newestId) unpinFromNewest();
      setSettledResultId(id);
    },
    [newestId, unpinFromNewest],
  );
  useScrollToResultParam(
    feedRootRef,
    generations,
    resultId,
    reduceMotion,
    onResultScrolled,
  );

  const isEmpty = generations.length === 0 && !isLoading && !loadError;
  const isLoadError = Boolean(loadError) && generations.length === 0 && !isLoading;
  const itemMotion = useMemo(
    () =>
      reduceMotion
        ? { layout: false as const, initial: false as const }
        : {
            layout: "position" as const,
            initial: { opacity: 0 },
            animate: { opacity: 1 },
            transition: {
              layout: { duration: PUSH_S, ease: MOTION_EASE },
              opacity: { duration: 0.25, delay: REVEAL_DELAY_S, ease: MOTION_EASE },
            },
          },
    [reduceMotion],
  );

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
      <AnimatePresence initial={false}>
        {generations.map((generation) => (
          <motion.div
            key={generation.id}
            ref={generation.id === newestId ? newestRef : undefined}
            data-generation-id={generation.id}
            className={
              isInFlightGeneration(generation) ? styles.resultGroup : undefined
            }
            {...itemMotion}
          >
            <ResultFrameAdapter
              generation={generation}
              onCancel={onCancel}
              onRetry={onRetry}
              onDelete={onDelete}
              progressPercent={progressById.get(generation.id)}
            />
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
