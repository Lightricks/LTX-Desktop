import clsx from "clsx";

import { Flex } from "@ds/layout/Flex/Flex";

import styles from "./Progress.module.scss";

interface ProgressProps {
  value: number;
  className?: string;
}

export function Progress({ value, className }: ProgressProps) {
  const clampedValue = Math.min(100, Math.max(0, value));

  return (
    <Flex width="100%" height="fit-content" p="md" align="center" justify="center">
      <div className={styles.track}>
        <div
          className={clsx(styles.progress, className)}
          style={{ width: `${clampedValue}%` }}
        />
      </div>
    </Flex>
  );
}
