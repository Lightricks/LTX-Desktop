import { Search } from "lucide-react";

import { Button } from "@/ds/Button/Button";
import { Text } from "@ds/Text/Text";

import styles from "./LtxioQuickSearch.module.scss";

export function QuickSearchNoResults({ onClear }: { onClear: () => void }) {
  return (
    <div className={styles.empty}>
      <div className={styles.emptyStack}>
        <span className={styles.emptyIcon} aria-hidden>
          <Search strokeWidth={2} />
        </span>
        <div className={styles.emptyCopy}>
          <Text as="h3" variant="heading" size="md" align="center">
            No matches found
          </Text>
          <Text
            as="p"
            variant="body"
            size="md"
            align="center"
            className={styles.emptyBody}
          >
            We couldn&apos;t find anything matching your search. Try a different
            keyword, or clear search to browse.
          </Text>
        </div>
        <Button
          appearance="neutral"
          hierarchy="secondary"
          size="md"
          label="Clear search"
          onMouseDown={(event) => event.preventDefault()}
          onClick={onClear}
        />
      </div>
    </div>
  );
}
