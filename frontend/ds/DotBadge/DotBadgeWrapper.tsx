import { clsx } from "clsx";
import { forwardRef } from "react";

import { DotBadge } from "./DotBadge";
import styles from "./DotBadgeWrapper.module.scss";

export interface DotBadgeWrapperProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Whether to show the DotBadge at the top-right corner */
  showDotBadge?: boolean;
  children: React.ReactNode;
  className?: string;
}

/**
 * DotBadgeWrapper positions a DotBadge at the top-right corner of its children.
 * Use this for icon-only buttons, toolbar items, or any component where the dot
 * should overlay rather than sit inline.
 *
 * @example
 * ```tsx
 * <DotBadgeWrapper showDotBadge>
 *   <Button isIconOnly leftIcon={<EditIcon />} aria-label="Edit" />
 * </DotBadgeWrapper>
 * ```
 */
export const DotBadgeWrapper = forwardRef<HTMLDivElement, DotBadgeWrapperProps>(
  function DotBadgeWrapper({ showDotBadge, children, className, ...rest }, ref) {
    return (
      <div ref={ref} className={clsx(styles.wrapper, className)} {...rest}>
        {children}
        {showDotBadge && <DotBadge className={styles.dot} />}
      </div>
    );
  },
);
