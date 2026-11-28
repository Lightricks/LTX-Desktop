import { Text } from "@ds/Text/Text";

import styles from "./LtxioQuickSearch.module.scss";

function Hint({ keys, label }: { keys: string; label: string }) {
  return (
    <span className={styles.footerHint}>
      <kbd className={styles.kbd}>{keys}</kbd>
      <Text as="span" variant="body" size="sm" className={styles.footerHintLabel}>
        {label}
      </Text>
    </span>
  );
}

export function QuickSearchShortcutHints() {
  return (
    <div className={styles.footer}>
      <Hint keys="↵" label="Open" />
      <Hint keys="↑↓" label="Navigate" />
      <Hint keys="Esc" label="Close" />
    </div>
  );
}
