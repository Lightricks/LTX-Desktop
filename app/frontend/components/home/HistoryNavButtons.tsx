import { Button } from "@/ds/Button/Button";
import ArrowForwardIcon from "@/ds/assets/Icons/Arrow/Forward/Small.svg?react";

import styles from "./PageHeader.module.scss";
import { useHistoryNav } from "./useHistoryNav";

export function HistoryNavButtons() {
  const { canGoBack, canGoForward, goBack, goForward } = useHistoryNav();

  return (
    <div className={styles.historyNav}>
      <Button
        appearance="neutral"
        hierarchy="plain"
        size="md"
        isIconOnly
        disabled={!canGoBack}
        aria-label="Back"
        leftIcon={<ArrowForwardIcon className={styles.historyBack} />}
        onClick={goBack}
      />
      <Button
        appearance="neutral"
        hierarchy="plain"
        size="md"
        isIconOnly
        disabled={!canGoForward}
        aria-label="Forward"
        leftIcon={<ArrowForwardIcon />}
        onClick={goForward}
      />
    </div>
  );
}
