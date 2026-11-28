import { forwardRef } from "react";

import { Text } from "@ds/Text/Text";
import { Flex } from "@ds/layout/Flex/Flex";

import styles from "./ItemUserDetails.module.scss";

export interface ItemUserDetailsProps {
  /** The user's display name */
  name?: string;
  /** The user's email address */
  email?: string;
}

// Ensure at least one prop is provided
export type ItemUserDetailsPropsRequired =
  | { name: string; email?: string }
  | { name?: string; email: string };

/**
 * ItemUserDetails is a sub-component that should only be used within `DropdownMenu`.
 *
 * @see {@link ItemUserDetailsProps} for available props
 */
export const ItemUserDetails = forwardRef<
  HTMLDivElement,
  ItemUserDetailsPropsRequired
>((props, ref) => {
  const { name, email } = props;

  return (
    <Flex
      ref={ref}
      direction="column"
      p="md"
      className={styles.itemUserDetails}
    >
      {name && (
        <Text variant="label" size="md">
          {name}
        </Text>
      )}
      {email && (
        <Text variant="body" size="sm">
          {email}
        </Text>
      )}
    </Flex>
  );
});

ItemUserDetails.displayName = "ItemUserDetails";
