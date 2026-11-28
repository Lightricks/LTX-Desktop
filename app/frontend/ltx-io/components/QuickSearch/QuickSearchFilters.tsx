import { Text } from "@ds/Text/Text";

import styles from "./LtxioQuickSearch.module.scss";

export type QuickSearchFilter = {
  id: string;
  label: string;
};

export function QuickSearchFilters({
  filters,
  onJump,
}: {
  filters: readonly QuickSearchFilter[];
  onJump: (id: string) => void;
}) {
  if (filters.length === 0) return null;
  return (
    <div className={styles.keywords}>
      {filters.map((filter) => (
        <button
          key={filter.id}
          type="button"
          className={styles.keyword}
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => onJump(filter.id)}
        >
          <Text as="span" variant="body" size="md" className={styles.keywordLabel}>
            {filter.label}
          </Text>
        </button>
      ))}
    </div>
  );
}
