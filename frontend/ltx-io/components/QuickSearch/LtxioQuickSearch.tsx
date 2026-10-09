import * as Dialog from "@radix-ui/react-dialog";
import { clsx } from "clsx";
import { Search } from "lucide-react";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import CloseIcon from "@ds/assets/Icons/Close/Normal.svg?react";
import { Button } from "@/ds/Button/Button";
import { useKeyboardShortcut } from "@ds/lib/useKeyboardShortcut";
import { useThemedPortalContainer } from "@ds/styles/themes/useTheme";
import { useRecentSidebarTools } from "@/components/home/useRecentSidebarTools";

import type { QuickSearchNavigateOptions } from "./quickSearchNavigation.ts";

import { useQuickSearch } from "./QuickSearchContext.tsx";
import { QuickSearchEmptyState } from "./QuickSearchEmptyState.tsx";
import { QuickSearchFilters } from "./QuickSearchFilters.tsx";
import {
  QuickSearchFeaturePreview,
  QuickSearchPagePreview,
} from "./QuickSearchPreview.tsx";
import { QuickSearchNoResults } from "./QuickSearchNoResults.tsx";
import { QuickSearchResults } from "./QuickSearchResults.tsx";
import { QuickSearchShortcutHints } from "./QuickSearchShortcutHints.tsx";
import styles from "./LtxioQuickSearch.module.scss";
import {
  browseSectionsToRows,
  buildQuickSearchBrowseSections,
  defaultQuickSearchActiveIndex,
  searchQuickSearchRows,
  type QuickSearchRow,
} from "./quickSearchModel.ts";

export function LtxioQuickSearch({
  onNavigate,
}: {
  onNavigate: (path: string, options: QuickSearchNavigateOptions) => void;
}) {
  const portalContainer = useThemedPortalContainer();
  const { isOpen, close, toggle } = useQuickSearch();
  const recentFeatureIds = useRecentSidebarTools();
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const stickActiveItem = useRef(false);
  const trimmedQuery = query.trim();

  const browseSections = useMemo(
    () => buildQuickSearchBrowseSections(recentFeatureIds),
    [recentFeatureIds],
  );

  const rows = useMemo(() => {
    if (trimmedQuery) return searchQuickSearchRows(trimmedQuery);
    return browseSectionsToRows(browseSections);
  }, [browseSections, trimmedQuery]);

  const filters = useMemo(
    () =>
      trimmedQuery
        ? []
        : browseSections.map((section) => ({
            id: section.id,
            label: section.label,
          })),
    [browseSections, trimmedQuery],
  );

  useLayoutEffect(() => {
    if (!isOpen) return;
    setQuery("");
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const frame = window.requestAnimationFrame(() => inputRef.current?.focus());
    return () => window.cancelAnimationFrame(frame);
  }, [isOpen]);

  const defaultIndex = defaultQuickSearchActiveIndex(rows);
  useEffect(() => {
    setActiveIndex(defaultIndex);
  }, [defaultIndex, isOpen, trimmedQuery]);

  useEffect(() => {
    if (!stickActiveItem.current) return;
    stickActiveItem.current = false;
    resultsRef.current
      ?.querySelector<HTMLElement>('[aria-selected="true"]')
      ?.scrollIntoView({ block: "nearest" });
  }, [activeIndex]);

  const shouldHandleSlash = useCallback(
    (event: KeyboardEvent) =>
      !event.metaKey &&
      !event.ctrlKey &&
      (event.key === "/" || event.code === "Slash" || event.code === "NumpadDivide"),
    [],
  );

  useKeyboardShortcut({
    shouldHandle: shouldHandleSlash,
    onDown: (event) => {
      event.preventDefault();
      toggle();
    },
    options: { preventInInputs: true },
  });

  const openFeature = (path: string) => {
    onNavigate(path, { clearActiveProject: true });
    close();
  };

  const openPage = (path: string, clearsProject: boolean) => {
    onNavigate(path, { clearActiveProject: clearsProject });
    close();
  };

  const activateRow = (row: QuickSearchRow) => {
    if (row.kind === "feature") {
      openFeature(row.feature.path);
      return;
    }
    openPage(row.destination.path, row.destination.id === "page-home");
  };

  const scrollToSection = (sectionId: string) => {
    const scroller = resultsRef.current;
    if (!scroller) return;
    const target = scroller.querySelector<HTMLElement>(
      `[data-section="${sectionId}"]`,
    );
    if (!target) return;
    const paddingTop = Number.parseFloat(getComputedStyle(scroller).paddingTop) || 0;
    const top =
      scroller.scrollTop +
      target.getBoundingClientRect().top -
      scroller.getBoundingClientRect().top -
      paddingTop;
    scroller.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
    const firstInSection = rows.findIndex((row) => row.sectionId === sectionId);
    if (firstInSection >= 0) setActiveIndex(firstInSection);
  };

  const moveHighlight = (direction: 1 | -1) => {
    if (rows.length === 0) return;
    stickActiveItem.current = true;
    setActiveIndex((index) => (index + direction + rows.length) % rows.length);
  };

  const activeRow = rows[activeIndex] ?? null;
  const showEmptySearch = trimmedQuery.length > 0 && rows.length === 0;
  const hasActiveSearch = query.length > 0;
  const showPills = !trimmedQuery && filters.length > 0;

  const onInputKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      moveHighlight(1);
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      moveHighlight(-1);
      return;
    }
    if (event.key === "Enter" && activeRow != null) {
      event.preventDefault();
      activateRow(activeRow);
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      if (hasActiveSearch) {
        setQuery("");
        return;
      }
      close();
    }
  };

  const clearSearch = () => {
    setQuery("");
    inputRef.current?.focus();
  };

  const onCloseOrClear = () => {
    if (hasActiveSearch) {
      clearSearch();
      return;
    }
    close();
  };

  const previewPane =
    activeRow?.kind === "feature" ? (
      <QuickSearchFeaturePreview
        key={activeRow.id}
        feature={activeRow.feature}
        onOpen={() => activateRow(activeRow)}
        onPointerInteractionEnd={() => inputRef.current?.focus()}
      />
    ) : activeRow?.kind === "page" ? (
      <QuickSearchPagePreview
        key={activeRow.id}
        destination={activeRow.destination}
        onOpen={() => activateRow(activeRow)}
      />
    ) : (
      <QuickSearchEmptyState />
    );

  return (
    <Dialog.Root
      open={isOpen}
      onOpenChange={(next) => {
        if (!next) close();
      }}
    >
      <Dialog.Portal container={portalContainer ?? undefined}>
        <Dialog.Overlay className={styles.backdrop} />
        <Dialog.Content
          className={styles.dialog}
          aria-describedby={undefined}
          onOpenAutoFocus={(event) => event.preventDefault()}
        >
          <Dialog.Title className={styles.visuallyHidden}>Search</Dialog.Title>
          <div className={styles.inputRow}>
            <Search className={styles.searchIcon} aria-hidden strokeWidth={2} />
            <input
              ref={inputRef}
              className={styles.input}
              type="text"
              role="combobox"
              value={query}
              placeholder="Search…"
              autoComplete="off"
              spellCheck={false}
              aria-expanded
              aria-autocomplete="list"
              aria-controls="ltxio-quick-search-results"
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={onInputKeyDown}
            />
            <Button
              appearance="neutral"
              hierarchy="plain"
              size="sm"
              isIconOnly
              leftIcon={<CloseIcon />}
              className={styles.clear}
              aria-label={hasActiveSearch ? "Clear search" : "Close search"}
              onMouseDown={(event) => event.preventDefault()}
              onClick={onCloseOrClear}
            />
          </div>
          {showPills ? (
            <QuickSearchFilters filters={filters} onJump={scrollToSection} />
          ) : null}
          <div className={clsx(styles.body, showEmptySearch && styles.bodyEmpty)}>
            {showEmptySearch ? (
              <QuickSearchNoResults onClear={clearSearch} />
            ) : (
              <>
                <div className={styles.resultsPane}>
                  <div ref={resultsRef} className={styles.results}>
                    <QuickSearchResults
                      rows={rows}
                      query={query}
                      activeIndex={activeIndex}
                      onHover={setActiveIndex}
                      onSelect={activateRow}
                    />
                  </div>
                  <QuickSearchShortcutHints />
                </div>
                {previewPane}
              </>
            )}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
