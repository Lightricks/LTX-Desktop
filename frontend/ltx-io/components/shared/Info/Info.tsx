import { clsx } from "clsx";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import { Text } from "@ds/Text/Text";
import {
  TOOLTIP_DEFAULT_OPEN_DELAY_MS,
  Tooltip,
  type TooltipAlign,
  type TooltipSide,
} from "@ds/Tooltip/Tooltip";
import InfoLineIcon from "@ds/assets/Icons/Info/Line.svg?react";

import styles from "./Info.module.scss";

export const INFO_SIZES = ["xs", "sm", "md", "lg", "xl"] as const;
export type InfoSize = (typeof INFO_SIZES)[number];

// Mirror the shared Tooltip open delay so hovering the label feels the same as
// hovering any other tooltip in the app.
export const INFO_DEFAULT_OPEN_DELAY_MS = TOOLTIP_DEFAULT_OPEN_DELAY_MS;
const DEFAULT_OPEN_DELAY_MS = INFO_DEFAULT_OPEN_DELAY_MS;

export function Info({
  label,
  content,
  side,
  align,
  sideOffset,
  alignOffset,
  maxWidth,
  className,
  size = "sm",
  delay,
}: {
  label?: string;
  content: ReactNode;
  side?: TooltipSide;
  align?: TooltipAlign;
  sideOffset?: number;
  alignOffset?: number;
  maxWidth?: number;
  className?: string;
  size?: InfoSize;
  delay?: number;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const openTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const cancelPendingOpen = useCallback(() => {
    if (openTimerRef.current !== null) {
      clearTimeout(openTimerRef.current);
      openTimerRef.current = null;
    }
  }, []);

  const showTooltip = useCallback(() => {
    cancelPendingOpen();
    const wait = delay ?? DEFAULT_OPEN_DELAY_MS;
    if (wait <= 0) {
      setIsOpen(true);
      return;
    }
    openTimerRef.current = setTimeout(() => setIsOpen(true), wait);
  }, [cancelPendingOpen, delay]);

  const hideTooltip = useCallback(() => {
    cancelPendingOpen();
    setIsOpen(false);
  }, [cancelPendingOpen]);

  useEffect(() => cancelPendingOpen, [cancelPendingOpen]);

  return (
    <span
      className={styles.container}
      onMouseEnter={showTooltip}
      onMouseLeave={hideTooltip}
      onFocus={showTooltip}
      onBlur={hideTooltip}
    >
      {label && (
        <Text variant="label" size={size} as="span">
          {label}
        </Text>
      )}
      <Tooltip
        content={content}
        side={side}
        align={align}
        sideOffset={sideOffset}
        alignOffset={alignOffset}
        maxWidth={maxWidth}
        className={className}
        open={isOpen}
        onOpenChange={setIsOpen}
      >
        <span className={styles.iconWrapper}>
          <InfoLineIcon className={clsx(styles.icon, styles[size])} />
        </span>
      </Tooltip>
    </span>
  );
}
