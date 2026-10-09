import { clsx } from "clsx";
import type { ReactNode } from "react";

import { Text } from "@ds/Text/Text";

import styles from "./ResultStatus.module.scss";

export function ResultStatus({
  icon,
  title,
  body,
  action,
  className,
  copyClassName,
}: {
  icon?: ReactNode;
  title?: string;
  body?: string;
  action?: ReactNode;
  className?: string;
  copyClassName?: string;
}) {
  const hasCopy = Boolean(title || body);

  return (
    <div className={clsx(styles.root, className)}>
      {icon !== undefined ? (
        <span className={styles.icon} aria-hidden>
          {icon}
        </span>
      ) : null}
      {hasCopy ? (
        <div className={clsx(styles.copy, copyClassName)}>
          {title ? (
            <Text
              as="span"
              variant="heading"
              size="md"
              align="center"
              className={styles.title}
            >
              {title}
            </Text>
          ) : null}
          {body ? (
            <Text as="p" variant="body" size="lg" align="center" className={styles.body}>
              {body}
            </Text>
          ) : null}
        </div>
      ) : null}
      {action}
    </div>
  );
}
