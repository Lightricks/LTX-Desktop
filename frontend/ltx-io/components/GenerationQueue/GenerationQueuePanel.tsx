import {
  DndContext,
  DragOverlay,
  KeyboardSensor,
  MouseSensor,
  TouchSensor,
  defaultAnnouncements,
  useSensor,
  useSensors,
  type DragEndEvent,
  type DragStartEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router";

import SectionVideosIcon from "@ds/assets/Icons/SectionVideos.svg?react";
import { animationProps } from "@ds/lib/popoverAnimationOptions";
import { Button } from "@/ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { Skeleton } from "@ds/Skeleton/Skeleton";
import { TooltipProvider } from "@ds/Tooltip/Tooltip";
import {
  generationQueueQueryOptions,
  useGenerationQueue,
} from "../../hooks/useGenerationQueue.ts";
import {
  invalidateAffectedGenerationQueries,
  replaceGenerationQueueCache,
  type ExploreQueueEntry,
  type ExploreQueueSnapshot,
  type GenerationQueueApi,
} from "../../hooks/generationQueryKeys.ts";
import { useReorderGenerationQueue } from "../../hooks/useReorderGenerationQueue.ts";
import { unwrapApiResult } from "../../lib/unwrapApiResult.ts";
import type { ExploreAsset } from "@/lib/explore-contract";
import { HOME_FEATURES } from "@/lib/home-features";
import type {
  ExploreApi,
  ExploreGenerationPollingPolicy,
} from "../../runtime/ExploreRuntime.tsx";
import {
  ActiveQueueRow,
  FinishedQueueRow,
  GenerationQueueDragPreview,
  GenerationQueueSortableRow,
} from "./GenerationQueueRow.tsx";
import {
  getQueuePanelState,
  isQueueReorderConflict,
  queuedEntryForActiveDrag,
  reorderQueuedEntries,
  requestForQueueOrder,
  resultLocationForFeature,
} from "./generationQueuePanelModel.ts";
import { useFinishedQueueEntries } from "./useFinishedQueueEntries.ts";
import { ResultStatus } from "../shared/ResultStatus/ResultStatus.tsx";
import styles from "./GenerationQueuePanel.module.scss";

export type GenerationQueuePanelProps = {
  api: GenerationQueueApi &
    Pick<
      ExploreApi,
      | "cancelQueuedGeneration"
      | "markGenerationQueueDoneSeen"
      | "dismissGenerationQueueDone"
      | "clearGenerationQueueDone"
      | "clearGenerationQueueFailed"
    >;
  generationPolling: ExploreGenerationPollingPolicy;
  mediaUrlForAsset: (asset: ExploreAsset) => string | null;
  thumbUrlForAsset: (asset: ExploreAsset) => string | null;
  placement?: "floating" | "header";
};

export function GenerationQueuePanel({
  api,
  generationPolling,
  mediaUrlForAsset,
  thumbUrlForAsset,
  placement = "floating",
}: GenerationQueuePanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const preferReducedMotion = useReducedMotion();
  const panelMotion = preferReducedMotion
    ? {
        initial: false,
        animate: { opacity: 1, scale: 1, y: 0 },
        exit: { opacity: 0 },
        transition: { duration: 0 },
      }
    : animationProps;
  const [feedback, setFeedback] = useState<string | null>(null);
  const [activeDragId, setActiveDragId] = useState<string | null>(null);
  const [activeDragWidth, setActiveDragWidth] = useState<number | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const activeDragIdRef = useRef<string | null>(null);
  const queueRowListMotion = useMemo(
    () =>
      preferReducedMotion
        ? { initial: false, exit: { opacity: 0 }, transition: { duration: 0 } }
        : {
            layout: activeDragId == null ? ("position" as const) : false,
            initial: { opacity: 0, y: 4 },
            animate: { opacity: 1, y: 0 },
            exit: {
              opacity: 0,
              y: -4,
              transition: { duration: 0.12, ease: [0.2, 0, 1, 0.9] },
            },
            transition: { duration: 0.1, ease: [0, 0, 0.38, 0.9] },
          },
    [preferReducedMotion, activeDragId],
  );

  const closePanel = () => {
    setIsOpen(false);
    setFeedback(null);
  };

  const setActiveDrag = (dragId: string | null) => {
    activeDragIdRef.current = dragId;
    setActiveDragId(dragId);
  };

  useEffect(() => {
    if (!isOpen) return;
    const onPointerDown = (event: PointerEvent) => {
      if (activeDragIdRef.current != null) return;
      const target = event.target;
      if (!(target instanceof Node)) return;
      const root = rootRef.current;
      if (root?.contains(target)) return;
      closePanel();
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      if (activeDragIdRef.current != null) return;
      event.preventDefault();
      closePanel();
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [isOpen]);
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const queueQuery = useGenerationQueue({
    api,
    generationPolling,
    isLive: isOpen,
  });
  const reorder = useReorderGenerationQueue({ api });
  const {
    doneEntries,
    failedEntries,
    unreadDoneIds,
    clearDone,
    clearFailed,
    removeDoneEntry,
    markDoneRead,
  } = useFinishedQueueEntries({
    snapshot: queueQuery.data,
    api,
  });

  const openDoneResult = (entry: ExploreQueueEntry) => {
    markDoneRead(entry.generation.id);
    const location = resultLocationForFeature(
      entry.generation.feature,
      entry.generation.id,
      HOME_FEATURES,
    );
    if (location == null) {
      setFeedback("Couldn’t open this result.");
      return;
    }
    closePanel();
    navigate(location);
  };

  const cancel = useMutation({
    mutationFn: async (generationId: string) =>
      unwrapApiResult(await api.cancelQueuedGeneration(generationId)),
    onSuccess: () => {
      invalidateAffectedGenerationQueries(queryClient);
    },
  });
  const cancelGeneration = (generationId: string) => {
    cancel.mutate(generationId, {
      onError: () => {
        setFeedback("Couldn’t cancel that generation. Try again.");
      },
    });
  };
  const sensors = useSensors(
    useSensor(MouseSensor, { activationConstraint: { distance: 8 } }),
    useSensor(TouchSensor, {
      activationConstraint: { delay: 150, tolerance: 5 },
    }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );
  const snapshot: ExploreQueueSnapshot = queueQuery.data ?? {
    active: null,
    queued: [],
    done: [],
    failed: [],
    unseen_ids: [],
  };
  const panelState = getQueuePanelState(snapshot, {
    isReordering: reorder.isPending,
    isCancelPending: cancel.isPending,
  });
  const draggingQueuedEntry = queuedEntryForActiveDrag(
    snapshot.queued,
    activeDragId,
  );

  const submitOrder = (
    queued: readonly ExploreQueueEntry[],
    generationId: string,
  ) => {
    const request = requestForQueueOrder(queued, generationId);
    if (request == null) return;
    setFeedback(null);
    reorder.mutate(request, {
      onError: (error) => {
        if (isQueueReorderConflict(error)) {
          setFeedback("Queue changed. Refreshed to the latest order.");
          void queryClient.fetchQuery(
            generationQueueQueryOptions({
              api,
              generationPolling,
              isLive: true,
            }),
          );
          return;
        }
        setFeedback("Couldn’t reorder the queue. Try again.");
        invalidateAffectedGenerationQueries(queryClient);
      },
    });
  };

  const clearActiveDrag = () => {
    setActiveDrag(null);
    setActiveDragWidth(null);
  };

  const handleDragStart = ({ active }: DragStartEvent) => {
    setActiveDrag(String(active.id));
    setActiveDragWidth(
      active.rect.current.initial?.width ??
        active.rect.current.translated?.width ??
        null,
    );
  };

  const handleDragEnd = ({ active, over }: DragEndEvent) => {
    if (panelState.sortingDisabled || over == null || active.id === over.id) {
      clearActiveDrag();
      return;
    }
    const finalQueued = reorderQueuedEntries(
      snapshot.queued,
      String(active.id),
      String(over.id),
    );
    replaceGenerationQueueCache(queryClient, {
      ...snapshot,
      queued: finalQueued,
    });
    clearActiveDrag();
    submitOrder(finalQueued, String(active.id));
  };

  const queueCount =
    snapshot.queued.length +
    (snapshot.active ? 1 : 0) +
    doneEntries.length +
    failedEntries.length;
  const queueToggleAriaLabel =
    queueCount > 0 ? `Queue, ${queueCount} generations` : "Queue";
  const queueLiveMessage = queueQuery.isLoading
    ? "Loading generation queue…"
    : queueQuery.isError
      ? "Couldn’t load the generation queue."
      : queueCount === 0
        ? "No generations are waiting."
        : snapshot.active != null && snapshot.queued.length > 0
          ? `Generating. ${snapshot.queued.length} queued.`
          : snapshot.active != null
            ? "Generating."
            : snapshot.queued.length > 0
              ? `${snapshot.queued.length} queued.`
              : failedEntries.length > 0 && doneEntries.length > 0
                ? `${doneEntries.length} finished. ${failedEntries.length} failed.`
                : failedEntries.length > 0
                  ? `${failedEntries.length} failed.`
                  : `${doneEntries.length} finished.`;
  const isQueueEmpty =
    !queueQuery.isLoading &&
    queueQuery.data != null &&
    snapshot.active == null &&
    snapshot.queued.length === 0 &&
    doneEntries.length === 0 &&
    failedEntries.length === 0;

  return (
    <div ref={rootRef} className={styles.root} data-placement={placement}>
      <Button
        label="Queue"
        appearance="neutral"
        hierarchy="secondary"
        aria-label={queueToggleAriaLabel}
        aria-expanded={isOpen}
        aria-controls="generation-queue-panel"
        rightIcon={
          queueCount > 0 ? (
            <Text
              as="span"
              variant="label"
              size="md"
              className={styles.queueToggleCount}
            >
              ({queueCount})
            </Text>
          ) : undefined
        }
        onClick={() => {
          setIsOpen((open) => {
            const nextOpen = !open;
            if (nextOpen) {
              void queryClient.fetchQuery(
                generationQueueQueryOptions({
                  api,
                  generationPolling,
                  isLive: true,
                }),
              );
            }
            return nextOpen;
          });
          setFeedback(null);
        }}
      />
      <AnimatePresence initial={false}>
        {isOpen ? (
          <motion.aside
            key="generation-queue-panel"
            id="generation-queue-panel"
            className={styles.panel}
            data-empty={isQueueEmpty ? "" : undefined}
            aria-label="Generation queue"
            style={{ transformOrigin: "top right" }}
            {...panelMotion}
          >
        <TooltipProvider delay={0}>
          <p className={styles.srOnly} aria-live="polite" aria-atomic="true">
            {isOpen ? queueLiveMessage : ""}
          </p>
          {feedback ? (
            <p className={styles.feedback} role="status">
              {feedback}
            </p>
          ) : null}
          {queueQuery.isError ? (
            <p className={styles.feedback} role="alert">
              Couldn’t load the generation queue.
            </p>
          ) : null}
          {queueQuery.isLoading && queueQuery.data == null ? (
            <ul className={styles.rows} aria-busy="true" aria-label="Loading queue">
              {[0, 1].map((key) => (
                <li key={key} className={styles.skeletonRow}>
                  <div className={styles.skeletonBlock}>
                    <Skeleton height={14} width="40%" />
                    <Skeleton height={64} width="100%" />
                    <Skeleton height={12} width="72%" />
                  </div>
                </li>
              ))}
            </ul>
          ) : isQueueEmpty ? (
            <ResultStatus
              className={styles.emptyState}
              icon={<SectionVideosIcon />}
              title="Nothing in the queue"
              body="No generations are waiting."
            />
          ) : (
          <DndContext
            sensors={sensors}
            accessibility={{ announcements: defaultAnnouncements }}
            onDragStart={handleDragStart}
            onDragEnd={handleDragEnd}
            onDragCancel={clearActiveDrag}
          >
            <div className={styles.listScroller}>
              <SortableContext
                items={panelState.sortableIds}
                strategy={verticalListSortingStrategy}
              >
                <ul className={styles.rows}>
                  <AnimatePresence initial={false} mode="popLayout">
                    {snapshot.active ? (
                      <ActiveQueueRow
                        key={snapshot.active.generation.id}
                        entry={snapshot.active}
                        isCancelling={
                          snapshot.active.generation.status === "cancelling"
                        }
                        cancelDisabled={panelState.activeCancelDisabled}
                        listItemMotion={queueRowListMotion}
                        mediaUrlForAsset={mediaUrlForAsset}
                        thumbUrlForAsset={thumbUrlForAsset}
                        onCancel={() =>
                          cancelGeneration(snapshot.active!.generation.id)
                        }
                      />
                    ) : null}
                    {snapshot.queued.length > 0 ? (
                      <li
                        key="queue-section-header"
                        className={styles.queueSectionHeader}
                        role="presentation"
                      >
                        <span className={styles.srOnly}>
                          {`Queue, ${snapshot.queued.length} waiting`}
                        </span>
                        <span aria-hidden>Queue</span>
                      </li>
                    ) : null}
                    {snapshot.queued.map((entry) => (
                      <GenerationQueueSortableRow
                        key={entry.generation.id}
                        entry={entry}
                        isCancelling={false}
                        cancelDisabled={panelState.queuedCancelDisabled}
                        isReordering={reorder.isPending}
                        listItemMotion={queueRowListMotion}
                        mediaUrlForAsset={mediaUrlForAsset}
                        thumbUrlForAsset={thumbUrlForAsset}
                        onCancel={() => cancelGeneration(entry.generation.id)}
                      />
                    ))}
                    {doneEntries.length > 0 ? (
                      <li
                        key="done-section-header"
                        className={styles.queueSectionHeader}
                        role="presentation"
                      >
                        <span className={styles.srOnly}>
                          {`Done, ${doneEntries.length} finished`}
                        </span>
                        <span aria-hidden>Done</span>
                        <button
                          type="button"
                          className={styles.sectionClear}
                          onClick={clearDone}
                        >
                          Clear all
                        </button>
                      </li>
                    ) : null}
                    {doneEntries.map((entry) => (
                      <FinishedQueueRow
                        key={entry.generation.id}
                        entry={entry}
                        listItemMotion={queueRowListMotion}
                        mediaUrlForAsset={mediaUrlForAsset}
                        thumbUrlForAsset={thumbUrlForAsset}
                        showUnreadDot={unreadDoneIds.has(entry.generation.id)}
                        onOpen={() => openDoneResult(entry)}
                        onRemove={() => removeDoneEntry(entry.generation.id)}
                      />
                    ))}
                    {failedEntries.length > 0 ? (
                      <li
                        key="failed-section-header"
                        className={styles.queueSectionHeader}
                        role="presentation"
                      >
                        <span className={styles.srOnly}>
                          {`Failed, ${failedEntries.length} failed`}
                        </span>
                        <span aria-hidden>Failed</span>
                        <button
                          type="button"
                          className={styles.sectionClear}
                          onClick={clearFailed}
                        >
                          Clear
                        </button>
                      </li>
                    ) : null}
                    {failedEntries.map((entry) => (
                      <FinishedQueueRow
                        key={entry.generation.id}
                        entry={entry}
                        failed
                        listItemMotion={queueRowListMotion}
                        mediaUrlForAsset={mediaUrlForAsset}
                        thumbUrlForAsset={thumbUrlForAsset}
                        onOpen={() => openDoneResult(entry)}
                      />
                    ))}
                  </AnimatePresence>
                </ul>
              </SortableContext>
            </div>
            <DragOverlay
              className={styles.dragOverlay}
              dropAnimation={null}
              style={
                activeDragWidth == null ? undefined : { width: activeDragWidth }
              }
            >
              {draggingQueuedEntry ? (
                <GenerationQueueDragPreview
                  entry={draggingQueuedEntry.entry}
                  mediaUrlForAsset={mediaUrlForAsset}
                  thumbUrlForAsset={thumbUrlForAsset}
                />
              ) : null}
            </DragOverlay>
          </DndContext>
          )}
        </TooltipProvider>
          </motion.aside>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
