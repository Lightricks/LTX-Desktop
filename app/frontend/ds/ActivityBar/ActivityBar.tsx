import { clsx } from "clsx";

import styles from "./ActivityBar.module.scss";

/**
 * A horizontal progress bar indicating remaining time or capacity.
 *
 * @component
 * @example
 * ```tsx
 * <ActivityBar progress={75} size="md" />
 * ```
 */
export function ActivityBar({
  progress,
  size = "sm",
  className,
}: {
  /** Current progress value (0–100) */
  progress: number;
  /** Size variant */
  size?: "sm" | "md" | "lg";
  /** Additional class name for the progress fill */
  className?: string;
}) {
  return (
    <div className={clsx(styles.container, styles[size])}>
      <div className={styles.bar} />
      <div
        className={clsx(styles.progress, className)}
        style={{
          width: `${progress}%`,
        }}
      ></div>
    </div>
  );
}
