import { useSortable } from "@dnd-kit/sortable";
import { AlertTriangle } from "lucide-react";
import { motion, type HTMLMotionProps } from "framer-motion";
import type { HTMLAttributes, KeyboardEvent, ReactNode, Ref } from "react";

import CloseIcon from "@ds/assets/Icons/Close/Normal.svg?react";
import { Badge } from "@ds/Badge/Badge";
import { Button } from "@/ds/Button/Button";
import { DotBadge } from "@ds/DotBadge/DotBadge";
import { Flex } from "@ds/layout/Flex/Flex";
import { Text } from "@ds/Text/Text";
import { Tooltip } from "@ds/Tooltip/Tooltip";

import type { ExploreQueueEntry } from "../../hooks/generationQueryKeys.ts";
import { generationPresentation } from "../../lib/generationPresentation.ts";
import { useGenerationNegativePrompt } from "../../hooks/useGenerationNegativePrompt.ts";
import type { ExploreAsset } from "@/lib/explore-contract";
import { HOME_FEATURES } from "@/lib/home-features";
import { DragGrip } from "../shared/DragGrip/DragGrip";
import { generationQueueTooltipPortal } from "./generationQueueTooltipPortal.ts";
import {
  activeQueueRowDetail,
  activeQueueTitleStatus,
  formatActiveQueueProgress,
  orderedQueueInputAssets,
  queueRowBadges,
  resultLocationForFeature,
} from "./generationQueuePanelModel.ts";
import { GenerationQueueInputAsset } from "./GenerationQueueInputAsset.tsx";
import styles from "./GenerationQueuePanel.module.scss";

export type GenerationQueueRowListMotion = Pick<
  HTMLMotionProps<"li">,
  "initial" | "animate" | "exit" | "transition" | "layout"
>;

type QueueRowShellProps = {
  entry: ExploreQueueEntry;
  mediaUrlForAsset: (asset: ExploreAsset) => string | null;
  thumbUrlForAsset: (asset: ExploreAsset) => string | null;
  titleSuffix?: ReactNode;
  showUnreadDot?: boolean;
  openable?: boolean;
  onOpen?: () => void;
};

type SortableBinding = Pick<
  ReturnType<typeof useSortable>,
  | "attributes"
  | "listeners"
  | "setNodeRef"
  | "setActivatorNodeRef"
  | "transform"
  | "transition"
  | "isDragging"
>;

function sortableTransformStyle(
  transform: SortableBinding["transform"],
): string | undefined {
  if (transform == null) return undefined;
  return `translate3d(${transform.x}px, ${transform.y}px, 0) scaleX(${transform.scaleX}) scaleY(${transform.scaleY})`;
}

function QueueGripIcon() {
  return <DragGrip />;
}

function QueueDragHandle({
  label,
  tooltip,
  disabled,
  buttonRef,
  dragHandleProps,
}: {
  label: string;
  tooltip: string;
  disabled?: boolean;
  buttonRef?: Ref<HTMLButtonElement>;
  dragHandleProps?: HTMLAttributes<HTMLElement>;
}) {
  return (
    <Tooltip
      content={tooltip}
      side="top"
      portalContainer={generationQueueTooltipPortal}
    >
      <button
        ref={buttonRef}
        type="button"
        className={styles.dragHandleButton}
        aria-label={label}
        disabled={disabled}
        {...dragHandleProps}
      >
        <QueueGripIcon />
      </button>
    </Tooltip>
  );
}

function pickQueuePreviewAsset(
  assets: ReturnType<typeof orderedQueueInputAssets>,
) {
  return (
    assets.find((asset) => asset.media_kind === "image") ??
    assets.find((asset) => asset.media_kind === "video") ??
    assets[0] ??
    null
  );
}

function openTargetKeyDown(onOpen: () => void) {
  return (event: KeyboardEvent) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    onOpen();
  };
}

function QueueRowShell({
  entry,
  mediaUrlForAsset,
  thumbUrlForAsset,
  titleSuffix,
  showUnreadDot = false,
  openable = false,
  onOpen,
}: QueueRowShellProps) {
  const presentation = generationPresentation(entry.generation);
  const negativePromptLine = useGenerationNegativePrompt(entry.generation);
  const hasPromptRow = presentation.prompt.length > 0 || negativePromptLine != null;
  const inputAssets = orderedQueueInputAssets(entry);
  const previewAsset = pickQueuePreviewAsset(inputAssets);
  const title = presentation.label;
  // A row on a provider lane says so. Without it, a later row can start first with no visible reason.
  const badges = queueRowBadges(presentation.badges, entry.generation.lane);

  // An image run has no Home screen to open on. Opening is the only way to read a done row,
  // so such a row shows no unread dot.
  const hasResultScreen =
    resultLocationForFeature(entry.generation.feature, entry.generation.id, HOME_FEATURES) != null;

  const layoutAttrs = {
    "data-has-asset": previewAsset != null ? "" : undefined,
    "data-has-prompt": hasPromptRow ? "" : undefined,
  };

  const body = (
    <>
      {previewAsset != null ? (
        <div className={styles.rowAsset}>
          <GenerationQueueInputAsset
            asset={previewAsset}
            mediaUrlForAsset={mediaUrlForAsset}
            thumbUrlForAsset={thumbUrlForAsset}
          />
        </div>
      ) : null}
      <div className={styles.rowHeader}>
        <div className={styles.rowTitle} title={title}>
          <Text as="span" variant="heading" size="sm" className={styles.rowTitleLabel}>
            {title}
          </Text>
          {titleSuffix}
          {showUnreadDot && hasResultScreen ? <DotBadge className={styles.doneUnreadDot} /> : null}
        </div>
        {badges.length > 0 || presentation.riskTooltip != null ? (
          <Flex
            gap="xxs"
            align="center"
            wrap="nowrap"
            className={styles.rowBadges}
          >
            {presentation.riskTooltip != null ? (
              <Tooltip
                content={presentation.riskTooltip}
                side="top"
                portalContainer={generationQueueTooltipPortal}
              >
                <span
                  className={styles.riskBadge}
                  aria-label={presentation.riskTooltip}
                >
                  <AlertTriangle aria-hidden />
                </span>
              </Tooltip>
            ) : null}
            {badges.map((badge) => (
              <Badge
                key={badge}
                appearance="overlay"
                size="xs"
                textVariant="body"
                text={badge}
              />
            ))}
          </Flex>
        ) : null}
      </div>
      {hasPromptRow ? (
        <div className={styles.promptRow}>
          {presentation.prompt.length > 0 ? (
            <Text
              as="p"
              variant="body"
              size="md"
              title={presentation.prompt}
              className={styles.prompt}
            >
              {presentation.prompt}
            </Text>
          ) : null}
          {negativePromptLine != null ? (
            <Text
              as="p"
              variant="body"
              size="md"
              title={negativePromptLine}
              className={styles.negativePrompt}
            >
              {negativePromptLine}
            </Text>
          ) : null}
        </div>
      ) : null}
    </>
  );

  if (openable && onOpen != null && hasResultScreen) {
    return (
      <div
        className={styles.rowOpenTarget}
        role="link"
        tabIndex={0}
        onClick={onOpen}
        onKeyDown={openTargetKeyDown(onOpen)}
        {...layoutAttrs}
      >
        {body}
      </div>
    );
  }

  return <>{body}</>;
}

/** The outer row grid. Matches the shell's own attributes, negative prompt line included. */
function useRowLayoutDataAttrs(entry: ExploreQueueEntry) {
  const presentation = generationPresentation(entry.generation);
  const negativePromptLine = useGenerationNegativePrompt(entry.generation);
  const previewAsset = pickQueuePreviewAsset(orderedQueueInputAssets(entry));
  return {
    "data-has-asset": previewAsset != null ? "" : undefined,
    "data-has-prompt":
      presentation.prompt.length > 0 || negativePromptLine != null ? "" : undefined,
  };
}

type ActiveQueueRowProps = {
  entry: ExploreQueueEntry;
  isCancelling: boolean;
  cancelDisabled: boolean;
  mediaUrlForAsset: (asset: ExploreAsset) => string | null;
  thumbUrlForAsset: (asset: ExploreAsset) => string | null;
  onCancel: () => void;
  onOpen: () => void;
  listItemMotion?: GenerationQueueRowListMotion;
};

export function ActiveQueueRow({
  entry,
  isCancelling,
  cancelDisabled,
  mediaUrlForAsset,
  thumbUrlForAsset,
  onCancel,
  onOpen,
  listItemMotion,
}: ActiveQueueRowProps) {
  const layoutAttrs = useRowLayoutDataAttrs(entry);
  const presentation = generationPresentation(entry.generation);
  const progress = entry.progress;
  const progressLabel =
    progress != null ? formatActiveQueueProgress(progress) : null;
  const apiStatusLabel = activeQueueTitleStatus(entry.api_status);
  const progressDetail = activeQueueRowDetail(progress, entry.api_status);
  const titleProgressSuffix = isCancelling ? (
    <Text
      as="span"
      variant="heading"
      size="sm"
      className={styles.rowTitleStatus}
    >
      {`\u00A0(Cancelling…)`}
    </Text>
  ) : progressLabel != null || apiStatusLabel != null ? (
    <Text
      as="span"
      variant="heading"
      size="sm"
      className={styles.rowTitleProgress}
    >
      {`\u00A0(${progressLabel ?? apiStatusLabel})`}
    </Text>
  ) : null;

  return (
    <motion.li
      {...listItemMotion}
      className={styles.row}
      data-active=""
    >
      <div
        className={styles.rowLayout}
        {...layoutAttrs}
        data-has-trailing=""
        data-has-drag-slot=""
      >
        <QueueRowShell
          entry={entry}
          mediaUrlForAsset={mediaUrlForAsset}
          thumbUrlForAsset={thumbUrlForAsset}
          titleSuffix={titleProgressSuffix}
          openable
          onOpen={onOpen}
        />
        <div className={styles.rowTrailing}>
          {entry.cancellable ? (
            <Tooltip
              content={
                isCancelling ? "Cancelling generation…" : "Cancel generation"
              }
              side="top"
              portalContainer={generationQueueTooltipPortal}
            >
              <Button
                appearance="neutral"
                hierarchy="secondary"
                size="md"
                isIconOnly
                leftIcon={<CloseIcon />}
                aria-label={
                  isCancelling
                    ? `Cancelling generation for ${presentation.label}`
                    : `Cancel generation for ${presentation.label}`
                }
                disabled={cancelDisabled}
                onClick={onCancel}
              />
            </Tooltip>
          ) : null}
        </div>
      </div>
      {isCancelling ? (
        <span className={styles.srOnly}>Cancelling generation</span>
      ) : progressLabel != null ? (
        <span className={styles.srOnly}>
          {progressDetail} — {progressLabel} complete
        </span>
      ) : apiStatusLabel != null ? (
        <span className={styles.srOnly}>{progressDetail}</span>
      ) : null}
    </motion.li>
  );
}

type FinishedQueueRowProps = {
  entry: ExploreQueueEntry;
  failed?: boolean;
  showUnreadDot?: boolean;
  mediaUrlForAsset: (asset: ExploreAsset) => string | null;
  thumbUrlForAsset: (asset: ExploreAsset) => string | null;
  onOpen: () => void;
  onRemove?: () => void;
  listItemMotion?: GenerationQueueRowListMotion;
};

export function FinishedQueueRow({
  entry,
  failed = false,
  showUnreadDot = false,
  mediaUrlForAsset,
  thumbUrlForAsset,
  onOpen,
  onRemove,
  listItemMotion,
}: FinishedQueueRowProps) {
  const layoutAttrs = useRowLayoutDataAttrs(entry);
  const presentation = generationPresentation(entry.generation);

  return (
    <motion.li {...listItemMotion} className={styles.row}>
      <div
        className={styles.rowLayout}
        {...layoutAttrs}
        data-has-trailing={onRemove != null ? "" : undefined}
      >
        <QueueRowShell
          entry={entry}
          mediaUrlForAsset={mediaUrlForAsset}
          thumbUrlForAsset={thumbUrlForAsset}
          showUnreadDot={showUnreadDot}
          openable
          onOpen={onOpen}
        />
        {onRemove != null ? (
          <div className={styles.rowTrailing}>
            <Tooltip
              content="Remove from Done"
              side="top"
              portalContainer={generationQueueTooltipPortal}
            >
              <Button
                appearance="neutral"
                hierarchy="secondary"
                size="md"
                isIconOnly
                leftIcon={<CloseIcon />}
                aria-label={`Remove ${presentation.label} from Done`}
                onClick={onRemove}
              />
            </Tooltip>
          </div>
        ) : null}
      </div>
      {failed ? (
        <span className={styles.srOnly}>Generation failed</span>
      ) : null}
    </motion.li>
  );
}

type QueuedSortableRowProps = {
  entry: ExploreQueueEntry;
  isCancelling: boolean;
  cancelDisabled: boolean;
  isReordering: boolean;
  mediaUrlForAsset: (asset: ExploreAsset) => string | null;
  thumbUrlForAsset: (asset: ExploreAsset) => string | null;
  onCancel: () => void;
  onOpen: () => void;
  listItemMotion?: GenerationQueueRowListMotion;
};

export function GenerationQueueSortableRow({
  entry,
  isCancelling,
  cancelDisabled,
  isReordering,
  mediaUrlForAsset,
  thumbUrlForAsset,
  onCancel,
  onOpen,
  listItemMotion,
}: QueuedSortableRowProps) {
  const layoutAttrs = useRowLayoutDataAttrs(entry);
  const presentation = generationPresentation(entry.generation);
  const sortable = useSortable({
    id: entry.generation.id,
    disabled: isReordering,
    animateLayoutChanges: () => false,
  });
  const isDragging = sortable.isDragging === true;
  const sortableStyle = {
    transform: sortableTransformStyle(sortable.transform ?? null),
    transition: sortable.transition,
  };

  return (
    <motion.li
      {...listItemMotion}
      className={styles.row}
      data-dragging={isDragging ? "" : undefined}
    >
      <div ref={sortable.setNodeRef} style={sortableStyle}>
        <div
          className={styles.rowLayout}
          {...layoutAttrs}
          data-has-trailing=""
          data-has-drag-slot=""
        >
          <QueueRowShell
            entry={entry}
            mediaUrlForAsset={mediaUrlForAsset}
            thumbUrlForAsset={thumbUrlForAsset}
            openable
            onOpen={onOpen}
          />
          <div className={styles.rowTrailing}>
            <Tooltip
              content={
                isCancelling ? "Cancelling generation…" : "Cancel generation"
              }
              side="top"
              portalContainer={generationQueueTooltipPortal}
            >
              <Button
                appearance="neutral"
                hierarchy="secondary"
                size="md"
                isIconOnly
                leftIcon={<CloseIcon />}
                aria-label={`Cancel generation for ${presentation.label}`}
                disabled={cancelDisabled}
                onClick={onCancel}
              />
            </Tooltip>
            <QueueDragHandle
              label={`Reorder ${presentation.label}`}
              tooltip="Drag to reorder."
              disabled={isReordering}
              buttonRef={sortable.setActivatorNodeRef}
              dragHandleProps={
                isReordering
                  ? undefined
                  : {
                      ...(sortable.attributes as HTMLAttributes<HTMLElement>),
                      ...(sortable.listeners as HTMLAttributes<HTMLElement>),
                    }
              }
            />
          </div>
        </div>
      </div>
    </motion.li>
  );
}

export function GenerationQueueDragPreview({
  entry,
  mediaUrlForAsset,
  thumbUrlForAsset,
}: {
  entry: ExploreQueueEntry;
  mediaUrlForAsset: (asset: ExploreAsset) => string | null;
  thumbUrlForAsset: (asset: ExploreAsset) => string | null;
}) {
  const layoutAttrs = useRowLayoutDataAttrs(entry);
  return (
    <div className={styles.row} data-overlay="" aria-hidden>
      <div
        className={styles.rowLayout}
        {...layoutAttrs}
        data-has-trailing=""
        data-has-drag-slot=""
      >
        <QueueRowShell
          entry={entry}
          mediaUrlForAsset={mediaUrlForAsset}
          thumbUrlForAsset={thumbUrlForAsset}
        />
        <div className={styles.rowTrailing}>
          <Button
            appearance="neutral"
            hierarchy="secondary"
            size="md"
            isIconOnly
            leftIcon={<CloseIcon />}
            tabIndex={-1}
            aria-hidden
            aria-label="Cancel generation"
          />
          <span className={styles.dragHandleButton} aria-hidden>
            <QueueGripIcon />
          </span>
        </div>
      </div>
    </div>
  );
}
