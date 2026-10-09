import { clsx } from "clsx";
import React, { forwardRef } from "react";

import { Badge } from "@ds/Badge/Badge";
import { Text } from "@ds/Text/Text";
import { Box } from "@ds/layout/Box/Box";
import { Flex } from "@ds/layout/Flex/Flex";

import { Thumbnail, type ThumbnailProps } from "../../Thumbnail/Thumbnail";

import styles from "./ItemActionInformative.module.scss";

/** Base props shared between icon and thumbnail variants */
interface BaseItemActionInformativeProps {
  /** The label text for the menu item */
  label: string;
  /** Optional description text */
  description?: string;
  /** Whether the item is in a selected state */
  isSelected?: boolean;
  /** Whether the item is disabled */
  isDisabled?: boolean;
  /** Optional badge component to display in the label */
  labelBadge?: React.ReactElement<typeof Badge>;
  /** Optional badges to display top right */
  badgesList?: React.ReactElement<typeof Badge>[];
  /**
   * Click handler for the item.
   * Note: When used within RadixDropdownMenu, this is passed to the onSelect prop of RadixDropdownMenu.Item
   * and the onClick handler here won't be called directly.
   */
  onClick?: () => void;
}

/** Props when using an icon */
interface WithIconProps extends BaseItemActionInformativeProps {
  icon: React.ReactNode;
  thumbnail?: never;
}

/** Props when using a thumbnail */
interface WithThumbnailProps extends BaseItemActionInformativeProps {
  thumbnail: ThumbnailProps;
  icon?: never;
}

/** Props for the ItemActionInformative component */
export type ItemActionInformativeProps = WithIconProps | WithThumbnailProps;

/**
 * ItemActionInformative is a sub-component that should only be used within `DropdownMenu` or `DropdownPicker`.
 *
 * @see {@link ItemActionInformativeProps} for available props
 *
 * When used within RadixDropdownMenu, the onClick handler is passed to the onSelect prop
 * of RadixDropdownMenu.Item and the onClick handler here won't be called directly.
 */
export const ItemActionInformative = forwardRef<
  HTMLDivElement,
  ItemActionInformativeProps
>((props, ref) => {
  const {
    label,
    description,
    icon: Icon,
    thumbnail,
    isSelected,
    isDisabled,
    labelBadge,
    badgesList,
    onClick,
  } = props;

  return (
    <Flex
      ref={ref}
      align="center"
      gap="md"
      p="md"
      className={clsx(
        styles.itemActionInformative,
        isSelected && styles.active,
        isDisabled && styles.disabled,
      )}
      onClick={isDisabled || !onClick ? undefined : onClick}
    >
      <Box
        p="md"
        className={clsx(styles.mediaWrapper, thumbnail && styles.thumbnail)}
      >
        {thumbnail && <Thumbnail {...thumbnail} size="lg" />}
        {Icon}
      </Box>
      <Flex direction="column" gap="xxs" width="100%">
        <Flex justify="between">
          {label && (
            <Flex align="center" gap="md">
              <Text variant="label" size="md" className={styles.label}>
                {label}
              </Text>
              {labelBadge && labelBadge}
            </Flex>
          )}
          <Flex align="center" gap="xxs" width="fit-content">
            {badgesList &&
              badgesList.length > 0 &&
              badgesList.map((badge, index) => (
                <React.Fragment key={`badge-${index}`}>{badge}</React.Fragment>
              ))}
          </Flex>
        </Flex>
        {description && (
          <Text variant="body" size="sm">
            {description}
          </Text>
        )}
      </Flex>
    </Flex>
  );
});

ItemActionInformative.displayName = "ItemActionInformative";
