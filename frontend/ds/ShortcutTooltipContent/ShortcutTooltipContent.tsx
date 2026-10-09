import { ReactNode } from "react";

import { Text } from "@ds/Text/Text";
import { Flex } from "@ds/layout/Flex/Flex";

import styles from "./ShortcutTooltipContent.module.scss";

export function ShortcutTooltipContent({
  title,
  shortcut,
}: {
  title: string;
  shortcut?: string;
}): ReactNode {
  return (
    <Flex gap="md" align="center" wrap="nowrap" className={styles.container}>
      <Text variant="label" size="md" as="span">
        {title}
      </Text>
      {shortcut && (
        <Text variant="body" size="sm" as="span" className={styles.shortcut}>
          {shortcut}
        </Text>
      )}
    </Flex>
  );
}
