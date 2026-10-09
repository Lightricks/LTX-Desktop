import type { ReactNode } from "react";
import { Text } from "@/ds/Text/Text";
import styles from "./SettingsPageLayout.module.scss";

export function SettingsAnchorSection({
  title,
  headerTrailing,
  children,
}: {
  title: string;
  headerTrailing?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className={styles.modalSection}>
      <header className={styles.modalSectionHeader}>
        <div className={styles.sectionHeaderTop}>
          <Text
            as="h2"
            variant="heading"
            size="lg"
            className={styles.sectionHeaderTitle}
          >
            {title}
          </Text>
          {headerTrailing ? (
            <div className={styles.sectionHeaderTrailing}>{headerTrailing}</div>
          ) : null}
        </div>
      </header>
      <div className={styles.modalSectionBody}>{children}</div>
    </div>
  );
}
