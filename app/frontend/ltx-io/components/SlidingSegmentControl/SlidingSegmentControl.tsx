import { clsx } from "clsx";
import { motion } from "framer-motion";
import { useLayoutEffect, useRef, useState, type Key, type ReactNode } from "react";

import { Tooltip } from "@ds/Tooltip/Tooltip";

import styles from "./SlidingSegmentControl.module.scss";

export type SlidingSegmentSize = "sm" | "md";

export type SlidingSegment<T extends Key> = {
  value: T;
  label: ReactNode;
  disabled?: boolean;
  disabledTooltip?: ReactNode;
};

type Props<T extends Key> = {
  segments: SlidingSegment<T>[];
  selectedSegment: T;
  onSegmentChanged: (value: T) => void;
  size?: SlidingSegmentSize;
  fullWidth?: boolean;
  "aria-label"?: string;
  className?: string;
};

export function SlidingSegmentControl<T extends Key>({
  segments,
  selectedSegment,
  onSegmentChanged,
  size = "sm",
  fullWidth = false,
  "aria-label": ariaLabel,
  className,
}: Props<T>) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [thumb, setThumb] = useState<{ left: number; width: number } | null>(
    null,
  );

  useLayoutEffect(() => {
    const track = trackRef.current;
    if (!track) return;
    const measure = () => {
      const active = track.querySelector<HTMLElement>('[data-active="true"]');
      if (!active) return;
      const t = track.getBoundingClientRect();
      const a = active.getBoundingClientRect();
      setThumb({
        left: a.left - t.left - track.clientLeft + track.scrollLeft,
        width: a.width,
      });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(track);
    return () => ro.disconnect();
  }, [selectedSegment, segments]);

  return (
    <div
      ref={trackRef}
      className={clsx(
        styles.segment,
        styles[size],
        fullWidth && styles.fullWidth,
        className,
      )}
      role="tablist"
      aria-label={ariaLabel}
    >
      {thumb ? (
        <motion.span
          className={styles.segmentThumb}
          aria-hidden
          initial={false}
          animate={{ x: thumb.left, width: thumb.width }}
          transition={{ type: "spring", stiffness: 480, damping: 40 }}
        />
      ) : null}
      {segments.map((segment) => {
        const active = segment.value === selectedSegment;
        const disabled = Boolean(segment.disabled);
        return (
          <Tooltip
            key={segment.value}
            content={segment.disabledTooltip}
            delay={0}
            disabled={!disabled || !segment.disabledTooltip}
          >
            <button
              type="button"
              role="tab"
              aria-selected={active}
              aria-disabled={disabled || undefined}
              data-active={active}
              className={styles.segmentItem}
              onClick={() => {
                if (!disabled && !active) onSegmentChanged(segment.value);
              }}
            >
              <span className={styles.segmentLabel}>{segment.label}</span>
            </button>
          </Tooltip>
        );
      })}
    </div>
  );
}
