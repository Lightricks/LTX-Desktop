import { clsx } from "clsx";
import { ReactNode, useCallback, useState } from "react";

import { ActivityCircular } from "@ds/ActivityCircular/ActivityCircular";
import { Button } from "@ds/Button/Button";
import { ButtonSize } from "@ds/Button/types";
import { Text, type TextSize } from "@ds/Text/Text";
import { Flex } from "@ds/layout/Flex/Flex";
import NSFWIcon from "@ds/assets/Icons/EyesBlink.svg?react";
import UpgradeIcon from "@ds/assets/Icons/Flash/Line.svg?react";
import DeleteIcon from "@ds/assets/Icons/Remove.svg?react";
import RetryIcon from "@ds/assets/Icons/Reset.svg?react";
import MissingIcon from "@ds/assets/Icons/Unlink.svg?react";

import styles from "./ActivityIndicator.module.scss";

type ActivityProps = {
  type: "activity";
  size: Size;
  /** Progress value (0–100). Omit for an infinite spinner. */
  progress?: number;
  message: ReactNode;
  messageClassName?: string;
  disabled: never;
  showCtaButton: never;
  onRetry: never;
  ctaLabel?: never;
  onCancel: never;
  onFallback?: never;
  showIconOnErrorButton: never;
  onDelete: never;
  deleteLabel?: never;
};

type ErrorProps = {
  type: "error";
  size: Size;
  onRetry: () => void;
  ctaLabel?: string;
  disabled: boolean;
  showCtaButton: boolean;
  message: ReactNode;
  messageClassName?: string;
  progress: never;
  onCancel: never;
  onFallback?: never;
  showIconOnErrorButton: boolean;
  onDelete?: () => void;
  deleteLabel?: string;
};

type QueuedProps = {
  type: "queued";
  size: Size;
  onCancel: () => void;
  disabled: boolean;
  showCtaButton: boolean;
  message: never;
  messageClassName: never;
  progress: never;
  onRetry: never;
  ctaLabel?: never;
  onFallback?: never;
  showIconOnErrorButton: never;
  onDelete: never;
  deleteLabel?: never;
};

type NSFWProps = {
  type: "nsfw";
  onFallback?: () => void | Promise<void>;
  onRetry?: () => void;
  messageClassName: never;
  progress: never;
  ctaLabel?: never;
  disabled: never;
  showCtaButton: never;
  onCancel: never;
  message: never;
  showIconOnErrorButton: never;
  onDelete: never;
  deleteLabel?: never;
};

type UpgradeProps = {
  type: "upgrade";
  messageClassName: never;
  progress: never;
  onRetry: () => void;
  ctaLabel?: never;
  disabled: never;
  showCtaButton: never;
  onCancel: never;
  onFallback?: never;
  message: never;
  showIconOnErrorButton: never;
  onDelete: never;
  deleteLabel?: never;
};

type NotFound = {
  type: "not-found";
  size: Size;
  onDelete: () => void;
  onRetry: never;
  showCtaButton: boolean;
  messageClassName: never;
  progress: never;
  /** Label for the delete action button. Defaults to "Delete this". */
  deleteLabel?: string;
  ctaLabel?: never;
  disabled: never;
  onCancel: never;
  onFallback?: never;
  message: never;
  showIconOnErrorButton: never;
};

type ActivityIndicatorProps = {
  size: Size;
  presentOverlay?: boolean;
  fullScreen?: boolean;
  "data-testid"?: string;
} & Partial<
  ActivityProps | ErrorProps | QueuedProps | NSFWProps | UpgradeProps | NotFound
>;

type Params = {
  activityCircularDiameter: number;
  activityCircularStrokeWidth: number;
  buttonSize: ButtonSize;
  labelSize: string;
  headingSize: TextSize;
  bodySize: TextSize;
};

const sizeToParams = {
  sm: {
    activityCircularDiameter: 16,
    activityCircularStrokeWidth: 2,
    buttonSize: "sm",
    labelSize: styles.sm,
    headingSize: "xs",
    bodySize: "sm",
  },
  md: {
    activityCircularDiameter: 32,
    activityCircularStrokeWidth: 4,
    buttonSize: "md",
    labelSize: styles.md,
    headingSize: "sm",
    bodySize: "md",
  },
  lg: {
    activityCircularDiameter: 40,
    activityCircularStrokeWidth: 6,
    buttonSize: "lg",
    labelSize: styles.lg,
    headingSize: "md",
    bodySize: "lg",
  },
  xl: {
    activityCircularDiameter: 48,
    activityCircularStrokeWidth: 6,
    buttonSize: "lg",
    labelSize: styles.xl,
    headingSize: "md",
    bodySize: "lg",
  },
} as const satisfies Record<string, Params>;

type Size = keyof typeof sizeToParams;

/**
 * Multi-state activity indicator for loading, error, queued, NSFW, upgrade, and not-found states.
 *
 * @component
 * @example
 * ```tsx
 * <ActivityIndicator type="activity" size="md" message="Loading..." />
 * <ActivityIndicator type="error" size="md" message="Failed" onRetry={retry} />
 * ```
 */
export function ActivityIndicator({
  type = "activity",
  size,
  disabled = false,
  message,
  progress,
  onRetry,
  ctaLabel,
  deleteLabel,
  onCancel,
  onDelete,
  showCtaButton = true,
  presentOverlay = true,
  messageClassName,
  onFallback,
  showIconOnErrorButton = true,
  "data-testid": dataTestId,
  fullScreen = false,
}: ActivityIndicatorProps) {
  const [isFallbackLoading, setIsFallbackLoading] = useState(false);
  const handleFallbackClick = useCallback(async () => {
    if (!onFallback || isFallbackLoading) {
      return;
    }

    setIsFallbackLoading(true);
    try {
      await onFallback();
    } finally {
      setIsFallbackLoading(false);
    }
  }, [onFallback, isFallbackLoading]);

  let content;
  switch (type) {
    case "activity":
      content = (
        <>
          <ActivityCircular
            progress={progress}
            size={sizeToParams[size].activityCircularDiameter}
          />
          {message && (
            <span
              className={clsx(
                styles.label,
                sizeToParams[size].labelSize,
                messageClassName,
              )}
            >
              {message}
            </span>
          )}
        </>
      );
      break;

    case "error":
      content = (
        <>
          {message && (
            <span
              className={clsx(
                styles.label,
                sizeToParams[size].labelSize,
                messageClassName,
              )}
            >
              {message}
            </span>
          )}
          <Flex direction="row" gap="xs">
            {showCtaButton && onRetry && (
              <Button
                size={sizeToParams[size].buttonSize}
                appearance="overlay"
                hierarchy="primary"
                label={ctaLabel || "Retry"}
                leftIcon={showIconOnErrorButton ? <RetryIcon /> : undefined}
                onClick={onRetry}
                disabled={disabled}
              />
            )}
            {showCtaButton && onDelete && (
              <Button
                size={sizeToParams[size].buttonSize}
                appearance="overlay"
                hierarchy="primary"
                label={deleteLabel || "Delete"}
                leftIcon={<DeleteIcon />}
                onClick={onDelete}
              />
            )}
          </Flex>
        </>
      );
      break;

    case "queued":
      content = (
        <>
          <span className={clsx(styles.label, sizeToParams[size].labelSize)}>
            Queued
          </span>
          {showCtaButton && (
            <Button
              size={sizeToParams[size].buttonSize}
              appearance="overlay"
              hierarchy="primary"
              label="Cancel"
              onClick={onCancel}
            />
          )}
        </>
      );
      break;

    case "nsfw":
      content = (
        <Flex
          align="center"
          justify="center"
          gap="sm"
          p="md"
          direction="column"
          className={styles.nsfwContainer}
          width="100%"
          height="100%"
        >
          <NSFWIcon className={styles.icon} />
          <Text
            variant="heading"
            size={sizeToParams[size].headingSize}
            align="center"
          >
            Request couldn&apos;t be generated
          </Text>
          <Text
            variant="body"
            size={sizeToParams[size].bodySize}
            align="center"
          >
            {onFallback
              ? "This request does not comply with the selected model’s guidelines. You can edit your prompt or generate it using LTX-2.3 for broader compatibility."
              : "We weren't able to process this generation. Please try again or adjust your prompt."}
          </Text>
          {onFallback ? (
            <Button
              size={sizeToParams[size].buttonSize}
              appearance="overlay"
              hierarchy="primary"
              label="Try again with LTX-2.3"
              onClick={() => void handleFallbackClick()}
              isLoading={isFallbackLoading}
              className={styles.ctaButton}
            />
          ) : (
            onRetry && (
              <Button
                size={sizeToParams[size].buttonSize}
                appearance="overlay"
                hierarchy="primary"
                label="Try again"
                leftIcon={<RetryIcon />}
                onClick={onRetry}
                className={styles.ctaButton}
              />
            )
          )}
        </Flex>
      );
      break;

    case "upgrade":
      content = (
        <Flex
          align="center"
          justify="center"
          gap="sm"
          p="md"
          direction="column"
          className={styles.upgradeContainer}
          width="100%"
          height="100%"
        >
          <UpgradeIcon className={styles.icon} />
          <Text
            variant="heading"
            size={sizeToParams[size].headingSize}
            align="center"
          >
            Not enough credits
          </Text>
          <Text
            variant="body"
            size={sizeToParams[size].bodySize}
            align="center"
          >
            Upgrade to unlock more and generate without limits.
          </Text>
          <Button
            size={sizeToParams[size].buttonSize}
            appearance="overlay"
            hierarchy="primary"
            label="Upgrade"
            onClick={onRetry}
            disabled={disabled}
            className={styles.ctaButton}
          />
        </Flex>
      );
      break;

    case "not-found":
      content = (
        <Flex
          align="center"
          justify="center"
          gap="sm"
          p="md"
          direction="column"
          className={styles.upgradeContainer}
          width="100%"
          height="100%"
        >
          <MissingIcon className={styles.icon} />
          <Text
            variant="heading"
            size={sizeToParams[size].headingSize}
            align="center"
          >
            Deleted Asset
          </Text>
          <Text
            variant="body"
            size={sizeToParams[size].bodySize}
            align="center"
          >
            This asset has been deleted
          </Text>
          {showCtaButton && (
            <Button
              size={sizeToParams[size].buttonSize}
              appearance="overlay"
              hierarchy="primary"
              label={deleteLabel ?? "Delete this"}
              onClick={onDelete}
              disabled={disabled}
              className={styles.ctaButton}
            />
          )}
        </Flex>
      );
      break;
  }
  return (
    <div
      className={clsx(
        styles.container,
        presentOverlay && styles.overlayBackground,
        size === "sm" && styles.sm,
        fullScreen && styles.fullScreen,
      )}
      data-testid={dataTestId}
    >
      {content}
    </div>
  );
}
