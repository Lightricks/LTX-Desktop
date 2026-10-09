import { clsx } from "clsx";
import { forwardRef, isValidElement } from "react";

import { Text } from "@ds/Text/Text";

import styles from "./Badge.module.scss";

export const BADGE_APPEARANCES = [
  "neutral",
  "brand",
  "brandSecondary",
  "overlay",
  "overlayInverse",
  "inverse",
  "success",
  "warning",
  "danger",
  "stroke",
  "strokeBrand",
  "disabled",
  "monetization",
  "transparent",
] as const;

export type BadgeAppearance = (typeof BADGE_APPEARANCES)[number];

export const BADGE_SIZES = ["xs", "sm", "md", "lg"] as const;
export type BadgeSize = (typeof BADGE_SIZES)[number];

export type BadgeProps = {
  /** The visual appearance of the badge */
  appearance: BadgeAppearance;
  /** Size of the badge */
  size: BadgeSize;
  /** Text content */
  text?: string;
  /** Additional CSS class names */
  className?: string;
  /** Icon rendered before the text */
  leftIcon?: React.ReactNode;
  /** Icon rendered after the text */
  rightIcon?: React.ReactNode;
  /** Whether the badge is in a disabled visual state */
  disabled?: boolean;
  /** Text variant for the inner Text component */
  textVariant?: "labelCaps" | "label" | "body";
  /** Whether to truncate overflowing text */
  truncateText?: boolean;
};

/**
 * Badge is a non-interactive status indicator component.
 * For clickable badges, wrap Badge in a button element.
 *
 * @example
 * ```tsx
 * // Non-interactive badge
 * <Badge appearance="brand" size="md" text="New" />
 *
 * // Clickable badge - wrap in button
 * <button onClick={handleClick}>
 *   <Badge appearance="brand" size="md" text="Click me" />
 * </button>
 * ```
 */
export const Badge = forwardRef<HTMLDivElement, BadgeProps>(
  (
    {
      appearance,
      size,
      text,
      className: classNameProp,
      leftIcon,
      rightIcon,
      disabled,
      textVariant = "labelCaps",
      truncateText = false,
    },
    ref,
  ) => {
    const badgeClassName = clsx(
      styles.content,
      styles[size],
      styles[appearance],
      disabled && styles.disabled,
      classNameProp,
    );

    return (
      <div ref={ref} role="status" className={badgeClassName}>
        {leftIcon && (
          <div
            className={clsx(styles.icon, styles[size], {
              [styles.image]:
                isValidElement(leftIcon) && leftIcon.type === "img",
            })}
          >
            {leftIcon}
          </div>
        )}
        {text && (
          <Text
            variant={textVariant}
            size={size}
            className={styles.text}
            shouldBreakAllWords={truncateText}
            shouldTruncate={truncateText}
          >
            {text}
          </Text>
        )}
        {rightIcon && (
          <div className={clsx(styles.icon, styles[size])}>{rightIcon}</div>
        )}
      </div>
    );
  },
);

Badge.displayName = "Badge";
