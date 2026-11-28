import { clsx } from "clsx";

import styles from "./Skeleton.module.scss";

export interface SkeletonProps {
  className?: string;
  width?: number | string;
  height?: number | string;
  /** Overrides the default border-radius token */
  borderRadius?: number | string;
  aspectRatio?: string;
  "data-testid"?: string;
}

/**
 * Skeleton displays an animated shimmer placeholder while content is loading.
 *
 * @example
 * ```tsx
 * <Skeleton width={120} height={80} />
 * <Skeleton aspectRatio="16/9" className={styles.thumb} />
 * ```
 */
export function Skeleton({
  className,
  width,
  height,
  borderRadius,
  aspectRatio,
  "data-testid": dataTestId = "skeleton",
}: SkeletonProps) {
  return (
    <span
      aria-hidden="true"
      data-testid={dataTestId}
      className={clsx(styles.skeleton, className)}
      style={{ width, height, aspectRatio, borderRadius }}
    />
  );
}
