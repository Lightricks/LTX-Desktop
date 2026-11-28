import { clsx } from "clsx";
import React, { forwardRef } from "react";

import { Badge, type BadgeProps } from "@ds/Badge/Badge";
import { Text } from "@ds/Text/Text";
import {
  Thumbnail,
  type ThumbnailProps,
} from "@ds/Thumbnail/Thumbnail";
import { Flex } from "@ds/layout/Flex/Flex";
import CheckedIcon from "@ds/assets/Icons/Accept.svg?react";
import ArrowIcon from "@ds/assets/Icons/Arrow/Forward/Small.svg?react";

import styles from "./ItemAction.module.scss";

export interface ItemActionProps {
  /** The text content of the dropdown menu item */
  text: string;
  /** Optional native title attribute for the item text */
  textTitle?: string;
  /** Optional class name for the item text */
  textClassName?: string;
  /** Optional inline style for the item text */
  textStyle?: React.CSSProperties;
  /** The caption text content of the dropdown menu item */
  caption?: string;
  /** Optional icon component to display on the left side of the item text.*/
  leftIcon?: React.ReactNode;
  /** Optional Badge component to display on the right side of the item text.
   * @see {@link Badge} for more details on the Badge component
   */
  badge?: React.ReactElement<BadgeProps>;
  /** Optional text content to display on the right side */
  rightText?: string;
  /** Optional native title attribute for the right text */
  rightTextTitle?: string;
  /** Optional class name for the right text */
  rightTextClassName?: string;
  /** Optional CTA component to display on the right side */
  rightCta?: React.ReactNode;
  /** Optional thumbnail image configuration object */
  thumbnail?: ThumbnailProps;
  /** Whether the item is in an active state */
  isActive?: boolean;
  /** Whether the item is selected */
  isSelected?: boolean;
  /** Whether the item is selectable */
  isSelectable?: boolean;
  /** Whether the item is disabled */
  isDisabled?: boolean;
  /** Whether the item should show a submenu arrow */
  shouldShowSubmenuArrow?: boolean;
  /** Whether the item text should be truncated */
  shouldTruncateText?: boolean;
  /** Whether the item text should break all words */
  shouldBreakAllWords?: boolean;
  /** Additional class name for the item */
  className?: string;
  /** Text variant to use for the item text. Defaults to "label" */
  textVariant?: "label" | "body";
  /** Whether to show the selected tick/checkmark on the right instead of the left */
  selectionPosition?: "left" | "right";
  /**
   * Click handler for the item.
   * Note: When used within RadixDropdownMenu, this is passed to the onSelect prop of RadixDropdownMenu.Item
   * and the onClick handler here won't be called directly.
   */
  onClick?: () => void;
  /** Mouse enter handler for the item */
  onMouseEnter?: React.MouseEventHandler<HTMLDivElement>;
  /** Mouse leave handler for the item */
  onMouseLeave?: React.MouseEventHandler<HTMLDivElement>;
}

function LeftContent({
  isSelectable,
  isSelected,
  isDisabled,
  thumbnail,
  leftIcon,
  text,
  textTitle,
  textClassName,
  textStyle,
  caption,
  badge,
  shouldTruncateText,
  shouldBreakAllWords,
  textVariant = "label",
  selectionPosition,
}: ItemActionProps) {
  if (badge && badge.type !== Badge) {
    throw new Error("badge must be a valid Badge component");
  }

  return (
    <Flex gap="md" align="center" className={styles.leftSection}>
      {isSelectable && selectionPosition === "left" && (
        <Flex
          align="center"
          justify="center"
          className={styles.checkIconWrapper}
        >
          {isSelected && <CheckedIcon />}
        </Flex>
      )}
      {thumbnail && <Thumbnail {...thumbnail} isDisabled={isDisabled} />}
      {leftIcon}
      <Flex direction="column" gap="xxs">
        <Text
          variant={textVariant}
          size="md"
          as="span"
          shouldTruncate={shouldTruncateText}
          shouldBreakAllWords={shouldBreakAllWords}
          className={textClassName}
          style={textStyle}
          title={textTitle}
        >
          {text}
        </Text>
        {caption && (
          <Text variant="body" size="xs" as="span" shouldTruncate>
            {caption}
          </Text>
        )}
      </Flex>
      {badge}
    </Flex>
  );
}

function RightContent({
  rightText,
  rightTextTitle,
  rightTextClassName,
  rightCta,
  shouldShowSubmenuArrow,
  shouldTruncateText,
  textVariant = "label",
  selectionPosition,
  isSelected,
}: ItemActionProps) {
  return (
    <Flex gap="xs" align="center" ml="auto" className={styles.right}>
      {rightText && (
        <Text
          variant={textVariant}
          size="md"
          as="span"
          shouldTruncate={shouldTruncateText}
          className={clsx(styles.text, rightTextClassName)}
          title={rightTextTitle}
        >
          {rightText}
        </Text>
      )}
      {rightCta}
      {selectionPosition === "right" && isSelected ? (
        <CheckedIcon />
      ) : (
        shouldShowSubmenuArrow && <ArrowIcon />
      )}
    </Flex>
  );
}

/**
 * ItemAction is a sub-component that should only be used within `DropdownMenu` or `DropdownPicker`.
 *
 * @see {@link ItemActionProps} for available props
 *
 * When used within RadixDropdownMenu, the onClick handler is passed to the onSelect prop
 * of RadixDropdownMenu.Item and the onClick handler here won't be called directly.
 */
export const ItemAction = forwardRef<HTMLDivElement, ItemActionProps>(
  function ItemAction(props, ref) {
    const {
      isActive,
      isDisabled,
      isSelected,
      onClick,
      className,
      onMouseEnter,
      onMouseLeave,
    } = props;

    return (
      <Flex
        ref={ref}
        justify="between"
        p="md"
        className={clsx(
          styles.itemAction,
          isActive && styles.active,
          isDisabled && styles.disabled,
          className,
        )}
        onClick={isDisabled || !onClick ? undefined : onClick}
        aria-checked={isSelected}
        aria-disabled={isDisabled}
        data-testid="menu-item-action"
        data-active={isActive}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
        onMouseDown={(e) => {
          e.preventDefault();
          e.stopPropagation();
        }}
      >
        <LeftContent {...props} />
        <RightContent {...props} />
      </Flex>
    );
  },
);

ItemAction.displayName = "DropdownMenu.ItemAction";
