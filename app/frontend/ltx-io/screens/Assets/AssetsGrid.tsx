import { Skeleton } from "@ds/Skeleton/Skeleton";
import { useInfiniteScroll } from "@ds/lib/useInfiniteScroll";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useLayoutEffect, useState } from "react";

import type { ExploreListedAsset } from "@/lib/explore-contract";

import styles from "./AssetsLibraryScreen.module.scss";
import { MediaTile } from "./MediaTile";
import { ASSETS_SKELETON_COUNT } from "./assetsCopy";
import { computeGridScrollMargin } from "./assetsGridLayout";

const HOME_SCROLL_ROOT_SELECTOR = "[data-home-scroll-root]";
const GRID_GAP_PX = 24;
const MEDIUM_MIN_TILE_PX = 280;
const MAX_COLUMNS = 4;
const TILE_ASPECT = 16 / 9;

function mediumColumnCount(width: number): number {
  if (width <= 0) return 3;
  const columns = Math.floor((width + GRID_GAP_PX) / (MEDIUM_MIN_TILE_PX + GRID_GAP_PX));
  return Math.min(MAX_COLUMNS, Math.max(1, columns || 1));
}

function tileHeightFor(width: number, columns: number): number {
  const tileWidth = (width - GRID_GAP_PX * (columns - 1)) / columns;
  return tileWidth / TILE_ASPECT;
}

export function AssetsGrid({
  items,
  isLoading,
  hasNextPage,
  isFetchingNextPage,
  layoutRoot,
  onNextPage,
  onOpenAsset,
  onDeleteAsset,
  pendingDeleteAssetIds,
  scrollRoot,
  labelVerb,
}: {
  items: ExploreListedAsset[];
  isLoading: boolean;
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  layoutRoot: HTMLElement | null;
  onNextPage: () => void;
  onOpenAsset: (asset: ExploreListedAsset) => void;
  onDeleteAsset?: (assetId: string) => Promise<void>;
  pendingDeleteAssetIds?: ReadonlySet<string>;
  /** When set, the grid scrolls inside this element instead of the home page. */
  scrollRoot?: HTMLElement | null;
  labelVerb?: "Open" | "Select";
}) {
  const [scrollElement, setScrollElement] = useState<HTMLElement | null>(null);
  const [gridElement, setGridElement] = useState<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);
  const [scrollMargin, setScrollMargin] = useState(0);

  useLayoutEffect(() => {
    if (scrollRoot !== undefined) {
      setScrollElement(scrollRoot);
      return;
    }
    setScrollElement(document.querySelector<HTMLElement>(HOME_SCROLL_ROOT_SELECTOR));
  }, [scrollRoot]);

  useLayoutEffect(() => {
    if (!gridElement || !scrollElement) return;

    const syncLayout = () => {
      const nextWidth = gridElement.clientWidth;
      const nextMargin = computeGridScrollMargin(
        gridElement.getBoundingClientRect(),
        scrollElement.getBoundingClientRect(),
        scrollElement.scrollTop,
      );
      setWidth((prev) => (prev === nextWidth ? prev : nextWidth));
      setScrollMargin((prev) => (prev === nextMargin ? prev : nextMargin));
    };

    const observer = new ResizeObserver(syncLayout);
    observer.observe(gridElement);
    if (layoutRoot) observer.observe(layoutRoot);

    syncLayout();
    return () => observer.disconnect();
  }, [gridElement, scrollElement, layoutRoot]);

  const displayCount = isLoading ? ASSETS_SKELETON_COUNT : items.length;
  const columns = mediumColumnCount(width);
  const tileHeight = tileHeightFor(
    width > 0 ? width : columns * MEDIUM_MIN_TILE_PX,
    columns,
  );
  const rowCount = Math.max(1, Math.ceil(displayCount / columns));

  const itemsCount =
    isLoading && items.length === 0 ? ASSETS_SKELETON_COUNT : items.length;

  const { InfiniteScrollSentinel, InfiniteScrollLoadingMore } = useInfiniteScroll({
    scrollContainer: scrollElement,
    hasNextPage: !isLoading && hasNextPage,
    onNextPage,
    itemsCount,
    isFetchingNextPage,
    fillContainer: true,
  });

  const rowVirtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => scrollElement,
    estimateSize: () => tileHeight,
    overscan: 4,
    gap: GRID_GAP_PX,
    scrollMargin,
  });

  useLayoutEffect(() => {
    rowVirtualizer.measure();
  }, [rowVirtualizer, tileHeight, columns, scrollMargin]);

  const virtualRows = rowVirtualizer.getVirtualItems();
  const rowTemplate = `repeat(${columns}, minmax(0, 1fr))`;

  return (
    <div
      ref={setGridElement}
      className={styles.grid}
      style={{ height: rowVirtualizer.getTotalSize() }}
    >
      {virtualRows.map((virtualRow) => {
        const startIndex = virtualRow.index * columns;
        const cells = Array.from({ length: columns }, (_, columnIndex) => {
          const itemIndex = startIndex + columnIndex;
          if (itemIndex >= displayCount) return null;
          if (isLoading) {
            return (
              <Skeleton
                key={`skeleton-${itemIndex}`}
                aspectRatio="16 / 9"
                className={styles.tile}
              />
            );
          }
          const item = items[itemIndex];
          if (!item) return null;
          return (
            <div key={item.id} className={styles.tile}>
              <MediaTile
                asset={item}
                onOpen={() => onOpenAsset(item)}
                onDelete={onDeleteAsset}
                isDeletePending={pendingDeleteAssetIds?.has(item.id) ?? false}
                labelVerb={labelVerb}
              />
            </div>
          );
        });

        return (
          <div
            key={virtualRow.key}
            className={styles.row}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: virtualRow.size,
              transform: `translateY(${virtualRow.start - scrollMargin}px)`,
              gridTemplateColumns: rowTemplate,
            }}
          >
            {cells}
          </div>
        );
      })}
      <div
        style={{
          position: "absolute",
          top: rowVirtualizer.getTotalSize(),
          left: 0,
          width: "100%",
        }}
      >
        <InfiniteScrollSentinel />
        <InfiniteScrollLoadingMore />
      </div>
    </div>
  );
}
