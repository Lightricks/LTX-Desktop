import { clsx } from "clsx";

import styles from "./LoadingPulse.module.scss";

type LoadingPulseProps = {
  className?: string;
  "data-testid"?: string;
};

/** Tile fill for an active generation. Prefer over a skeleton or spinner. */
export function LoadingPulse({
  className,
  "data-testid": dataTestId = "generation-loading-pulse",
}: LoadingPulseProps) {
  return (
    <span
      aria-hidden="true"
      data-testid={dataTestId}
      className={clsx(styles.pulse, className)}
    />
  );
}
