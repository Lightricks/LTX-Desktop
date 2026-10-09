import { clsx } from "clsx";
import { Search } from "lucide-react";
import { type Ref, forwardRef } from "react";

import { Text } from "@/ds/Text/Text";

import styles from "./QuickSearchLaunchField.module.scss";
import {
  QUICK_SEARCH_PLACEHOLDER,
  QUICK_SEARCH_TOOL_PLACEHOLDERS,
} from "./quickSearchCopy.ts";
import { useTypewriterPlaceholder } from "./useTypewriterPlaceholder.ts";

type QuickSearchLaunchFieldProps = {
  hidden?: boolean;
  className?: string;
  onActivate: () => void;
};

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export const QuickSearchLaunchField = forwardRef(function QuickSearchLaunchField(
  { hidden = false, className, onActivate }: QuickSearchLaunchFieldProps,
  ref: Ref<HTMLButtonElement>,
) {
  const reduceMotion = prefersReducedMotion();
  const rotatingPlaceholder = useTypewriterPlaceholder(
    QUICK_SEARCH_TOOL_PLACEHOLDERS,
    !reduceMotion,
  );
  const placeholderText = reduceMotion ? QUICK_SEARCH_PLACEHOLDER : rotatingPlaceholder;

  return (
    <button
      ref={ref}
      type="button"
      className={clsx(styles.field, hidden && styles.hidden, className)}
      aria-label={`Open ${QUICK_SEARCH_PLACEHOLDER}`}
      onClick={onActivate}
    >
      <span className={styles.fieldInner} aria-hidden>
        <Search className={styles.searchIcon} aria-hidden strokeWidth={2} />
        <Text as="span" variant="body" size="lg" className={styles.placeholder}>
          {placeholderText}
        </Text>
      </span>
      <span className={styles.shortcut} aria-hidden>
        <Text as="span" variant="label" size="sm" className={styles.shortcutKey}>
          /
        </Text>
      </span>
    </button>
  );
});
