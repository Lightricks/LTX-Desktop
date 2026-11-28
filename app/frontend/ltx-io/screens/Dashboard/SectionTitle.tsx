import { Text } from "@/ds/Text/Text";

import { Info } from "../../components/shared/Info/Info.tsx";

import styles from "./DashboardScreen.module.scss";

export function SectionTitle({ title, info }: { title: string; info: string }) {
  return (
    <Text as="h2" variant="heading" size="sm" className={styles.panelTitle}>
      <span className={styles.titleWithInfo}>
        {title}
        <Info content={info} side="top" align="start" maxWidth={320} size="md" />
      </span>
    </Text>
  );
}
