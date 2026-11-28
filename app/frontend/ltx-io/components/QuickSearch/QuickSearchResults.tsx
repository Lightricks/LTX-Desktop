import { clsx } from "clsx";
import { Fragment, useCallback, useEffect, useRef, type ReactNode } from "react";

import { homeFeatureIcon } from "@/components/home/homeFeatureIcons";
import { CUSTOM_LORA_HOME_FEATURE } from "@/lib/custom-lora";
import { Text } from "@ds/Text/Text";

import styles from "./LtxioQuickSearch.module.scss";
import { HighlightedQuery } from "./quickSearchHighlight.tsx";
import {
  QUICK_SEARCH_HOVER_INTENT_MS,
  quickSearchHoverIntent,
  type QuickSearchPointer,
} from "./quickSearchHover.ts";
import { quickSearchIconForPageId } from "./quickSearchNavIcons.tsx";
import { quickSearchOptionId, type QuickSearchRow } from "./quickSearchModel.ts";

function groupRows(rows: readonly QuickSearchRow[]) {
  const groups: {
    section?: string;
    sectionId?: string;
    items: { row: QuickSearchRow; index: number }[];
  }[] = [];
  for (const [index, row] of rows.entries()) {
    const last = groups[groups.length - 1];
    if (
      last &&
      last.sectionId === row.sectionId &&
      last.section === row.section
    ) {
      last.items.push({ row, index });
      continue;
    }
    groups.push({
      section: row.section,
      sectionId: row.sectionId,
      items: [{ row, index }],
    });
  }
  return groups;
}

function rowTitleAndIcon(row: QuickSearchRow): { title: string; icon: ReactNode } {
  switch (row.kind) {
    case "feature":
      return { title: row.feature.title, icon: homeFeatureIcon(row.feature.id) };
    case "custom-lora":
      // A saved LoRA uses the icon of the Custom LoRA feature.
      return { title: row.lora.name, icon: homeFeatureIcon(CUSTOM_LORA_HOME_FEATURE.id) };
    case "page":
      return {
        title: row.destination.title,
        icon: quickSearchIconForPageId(row.destination.id),
      };
    default: {
      const unhandled: never = row;
      throw new Error(`Unhandled quick search row: ${JSON.stringify(unhandled)}`);
    }
  }
}

export function QuickSearchResults({
  rows,
  query,
  activeIndex,
  onHover,
  onSelect,
}: {
  rows: readonly QuickSearchRow[];
  query: string;
  activeIndex: number;
  onHover: (index: number) => void;
  onSelect: (row: QuickSearchRow) => void;
}) {
  const pointer = useRef<QuickSearchPointer>({ x: Number.NaN, y: Number.NaN });
  const pendingHover = useRef<number | null>(null);
  const hoverTimer = useRef<number | undefined>(undefined);

  const cancelPendingHover = useCallback(() => {
    window.clearTimeout(hoverTimer.current);
    hoverTimer.current = undefined;
    pendingHover.current = null;
  }, []);

  useEffect(() => cancelPendingHover, [cancelPendingHover, rows]);

  const onRowPointer = (index: number, point: QuickSearchPointer) => {
    const intent = quickSearchHoverIntent(pointer.current, point);
    if (intent === "ignore") return;
    pointer.current = point;

    if (index === activeIndex) {
      cancelPendingHover();
      return;
    }
    if (intent === "immediate") {
      cancelPendingHover();
      onHover(index);
      return;
    }
    if (pendingHover.current === index) return;
    cancelPendingHover();
    pendingHover.current = index;
    hoverTimer.current = window.setTimeout(() => {
      if (pendingHover.current !== index) return;
      pendingHover.current = null;
      onHover(index);
    }, QUICK_SEARCH_HOVER_INTENT_MS);
  };

  return (
    <div
      id="ltxio-quick-search-results"
      role="listbox"
      className={styles.resultsList}
      onMouseLeave={cancelPendingHover}
    >
      {groupRows(rows).map((group) => {
        const options = group.items.map(({ row, index }) => {
          const { title, icon } = rowTitleAndIcon(row);
          return (
            <button
              key={row.id}
              id={quickSearchOptionId(row.id)}
              type="button"
              role="option"
              aria-selected={index === activeIndex}
              tabIndex={-1}
              className={clsx(styles.item, index === activeIndex && styles.active)}
              onMouseDown={(event) => event.preventDefault()}
              onMouseEnter={(event) =>
                onRowPointer(index, { x: event.clientX, y: event.clientY })
              }
              onMouseMove={(event) =>
                onRowPointer(index, { x: event.clientX, y: event.clientY })
              }
              onClick={() => onSelect(row)}
            >
              <span className={styles.itemIcon} aria-hidden>
                {icon}
              </span>
              <Text
                as="span"
                variant="body"
                size="lg"
                shouldTruncate
                className={styles.title}
              >
                {query.trim() ? (
                  <HighlightedQuery text={title} query={query} />
                ) : (
                  title
                )}
              </Text>
              <span className={styles.affordance} aria-hidden>
                <kbd className={styles.kbdInline}>↵</kbd>
              </span>
            </button>
          );
        });

        if (!group.section) {
          return <Fragment key={group.items[0]?.row.id}>{options}</Fragment>;
        }

        return (
          <div
            key={group.sectionId ?? group.section}
            role="group"
            aria-label={group.section}
            className={styles.sectionGroup}
          >
            <Text
              as="div"
              variant="body"
              size="md"
              className={styles.sectionLabel}
              data-section={group.sectionId}
            >
              {group.section}
            </Text>
            {options}
          </div>
        );
      })}
    </div>
  );
}
