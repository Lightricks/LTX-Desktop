import { clsx } from "clsx";

import styles from "./DotsLoader.module.scss";

type DotsLoaderProps = {
  className?: string;
  "data-testid"?: string;
};

/**
 * Indeterminate four-dot sweep (same motion as LoRA trainer waits).
 * Neutral fg/bg mix — not brand blue. Copy around it carries % / ETA.
 */
export function DotsLoader({
  className,
  "data-testid": dataTestId = "dots-loader",
}: DotsLoaderProps) {
  return (
    <div
      className={clsx(styles.dots, className)}
      role="status"
      aria-label="Loading"
      data-testid={dataTestId}
    >
      <span className={styles.dot} />
      <span className={styles.dot} />
      <span className={styles.dot} />
      <span className={styles.dot} />
    </div>
  );
}
