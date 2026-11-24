import { forwardRef } from "react";

import { Text } from "@ds/Text/Text";
import { Box } from "@ds/layout/Box/Box";
import { Flex } from "@ds/layout/Flex/Flex";

import styles from "./ItemHeadline.module.scss";

export interface ItemHeadlineProps {
  /** The text content of the dropdown menu item */
  text: string;
  /** Optional icon component to display on the left side */
  icon?: React.FunctionComponent<React.SVGProps<SVGSVGElement>>;
}

/**
 * ItemHeadline is a sub-component that should only be used within `DropdownMenu` or `DropdownPicker`.
 *
 * @see {@link ItemHeadlineProps} for available props
 */
export const ItemHeadline = forwardRef<HTMLDivElement, ItemHeadlineProps>(
  (props, ref) => {
    const { text, icon: Icon } = props;

    return (
      <Flex
        ref={ref}
        align="center"
        gap="xs"
        p="md"
        className={styles.itemHeadline}
      >
        {Icon && (
          <Box>
            <Icon aria-hidden="true" className={styles.icon} />
          </Box>
        )}
        <Text variant="labelCaps" size="md">
          {text}
        </Text>
      </Flex>
    );
  },
);

ItemHeadline.displayName = "ItemHeadline";
