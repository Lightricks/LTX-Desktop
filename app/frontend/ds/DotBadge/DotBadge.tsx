import { clsx } from "clsx";
import { forwardRef } from "react";

import styles from "./DotBadge.module.scss";

export interface DotBadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  /** Additional CSS class names */
  className?: string;
}

/**
 * DotBadge is a minimal inline dot indicator used to draw attention
 * to an element, such as flagging new or unread content.
 *
 * @example
 * ```tsx
 * <DotBadge />
 *
 * <Text variant="label" size="sm">
 *   Notifications <DotBadge />
 * </Text>
 * ```
 */
export const DotBadge = forwardRef<HTMLSpanElement, DotBadgeProps>(function DotBadge(
  { className, ...rest },
  ref,
) {
  return (
    <span
      ref={ref}
      role="status"
      aria-label="new"
      className={clsx(styles.dotBadge, className)}
      {...rest}
    />
  );
});
