import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { type ReactNode, useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router";

import { paths } from "@/paths";

import { AssetsEmptyState } from "./AssetsEmptyState";
import { AssetsErrorState } from "./AssetsErrorState";
import { AssetsGrid } from "./AssetsGrid";
import styles from "./AssetsLibraryScreen.module.scss";
import { AssetsLightbox } from "./AssetsLightbox";
import { AssetsToolbar } from "./AssetsToolbar";
import {
  ASSETS_DELETE_ERROR_DESCRIPTION,
  ASSETS_DELETE_ERROR_TITLE,
  ASSETS_SEARCH_DEBOUNCE_MS,
} from "./assetsCopy";
import { resolveAssetsLibraryView } from "./assetsLibraryView";
import {
  type AssetsListFilters,
  lightboxAssetIdFromSearchParams,
  searchParamsFromLibraryState,
} from "./assetsSearchParams";
import { resolveLightboxDeepLinkAction } from "./lightboxCollection";
import { useAssetsList } from "./useAssetsList";
import { isAssetDeleteInUseError, useDeleteAsset } from "./useDeleteAsset";

const DEFAULT_FILTERS: AssetsListFilters = {
  mediaKind: null,
  sort: "created_at-desc",
  q: "",
};

function filtersAreActive(filters: AssetsListFilters): boolean {
  return (
    filters.mediaKind !== null || filters.sort !== "created_at-desc" || filters.q !== ""
  );
}

export function AssetsLibraryScreen() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const {
    data,
    error,
    isLoading,
    isFetchingNextPage,
    isFetchNextPageError,
    hasNextPage,
    fetchNextPage,
    refetch,
    filters,
  } = useAssetsList();
  const deleteAsset = useDeleteAsset();
  const [chromeElement, setChromeElement] = useState<HTMLDivElement | null>(null);

  const filtersRef = useRef(filters);
  filtersRef.current = filters;
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lightboxPushedRef = useRef(false);
  const [searchDraft, setSearchDraft] = useState(filters.q);
  const lightboxAssetId = lightboxAssetIdFromSearchParams(searchParams);

  const writeLibraryState = (
    nextFilters: AssetsListFilters,
    nextLightboxAssetId: string | null,
    history: "push" | "replace",
  ) => {
    setSearchParams(searchParamsFromLibraryState(nextFilters, nextLightboxAssetId), {
      replace: history === "replace",
    });
  };

  const writeFilters = (next: AssetsListFilters) => {
    writeLibraryState(next, lightboxAssetId, "replace");
  };

  const openLightbox = (assetId: string) => {
    lightboxPushedRef.current = true;
    writeLibraryState(filtersRef.current, assetId, "push");
  };

  const selectLightboxAsset = (assetId: string) => {
    writeLibraryState(filtersRef.current, assetId, "replace");
  };

  const closeLightbox = () => {
    if (lightboxPushedRef.current) {
      lightboxPushedRef.current = false;
      navigate(-1);
      return;
    }
    writeLibraryState(filtersRef.current, null, "replace");
  };

  useEffect(() => {
    setSearchDraft(filters.q);
  }, [filters.q]);

  useEffect(() => {
    return () => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    };
  }, []);

  const items = data?.pages.flatMap((page) => page.items) ?? [];
  const lightboxIndex =
    lightboxAssetId === null
      ? -1
      : items.findIndex((item) => item.id === lightboxAssetId);

  useEffect(() => {
    if (lightboxAssetId === null) {
      lightboxPushedRef.current = false;
    }
  }, [lightboxAssetId]);

  useEffect(() => {
    const action = resolveLightboxDeepLinkAction({
      lightboxAssetId,
      lightboxIndex,
      isLoading,
      isFetchingNextPage,
      isFetchNextPageError,
      hasNextPage: Boolean(hasNextPage),
    });
    switch (action) {
      case "idle":
        return;
      case "fetchNextPage":
        void fetchNextPage();
        return;
      case "close":
        closeLightbox();
        return;
      default: {
        const _exhaustive: never = action;
        return _exhaustive;
      }
    }
  }, [
    fetchNextPage,
    hasNextPage,
    isFetchNextPageError,
    isFetchingNextPage,
    isLoading,
    lightboxAssetId,
    lightboxIndex,
  ]);

  useEffect(() => {
    if (
      items.length === 0 &&
      hasNextPage &&
      !isLoading &&
      !isFetchingNextPage &&
      !error
    ) {
      void fetchNextPage();
    }
  }, [error, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading, items.length]);

  const view = resolveAssetsLibraryView({
    isLoading,
    error,
    loadedItemCount: items.length,
    isFetchNextPageError,
    hasNextPage: Boolean(hasNextPage),
  });

  const handleSearchDraftChange = (next: string) => {
    setSearchDraft(next);
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    searchTimerRef.current = setTimeout(() => {
      searchTimerRef.current = null;
      writeFilters({ ...filtersRef.current, q: next });
    }, ASSETS_SEARCH_DEBOUNCE_MS);
  };

  const handleClearFilters = () => {
    if (searchTimerRef.current) {
      clearTimeout(searchTimerRef.current);
      searchTimerRef.current = null;
    }
    setSearchDraft("");
    writeFilters(DEFAULT_FILTERS);
  };

  const handleRetry = () => {
    switch (view.retryTarget) {
      case "fetchNextPage":
        void fetchNextPage();
        return;
      case "refetch":
        void refetch();
        return;
      default: {
        const _exhaustive: never = view.retryTarget;
        return _exhaustive;
      }
    }
  };

  let body: ReactNode;
  switch (view.kind) {
    case "error":
      body = (
        <div className={styles.status}>
          <AssetsErrorState onRetry={handleRetry} />
        </div>
      );
      break;
    case "empty":
      body = (
        <div className={styles.status}>
          <AssetsEmptyState
            filtered={filtersAreActive(filters)}
            onGoHome={() => navigate(paths.home)}
            onClearFilters={handleClearFilters}
          />
        </div>
      );
      break;
    case "loading":
    case "content":
      body = (
        <AssetsGrid
          items={items}
          isLoading={isLoading && items.length === 0}
          hasNextPage={Boolean(hasNextPage) && view.inlineError === null}
          isFetchingNextPage={isFetchingNextPage}
          layoutRoot={chromeElement}
          onNextPage={() => {
            void fetchNextPage();
          }}
          onOpenAsset={(asset) => openLightbox(asset.id)}
          onDeleteAsset={(assetId) => deleteAsset.mutateAsync(assetId)}
          pendingDeleteAssetIds={deleteAsset.pendingAssetIds}
        />
      );
      break;
    default: {
      const _exhaustive: never = view.kind;
      body = _exhaustive;
    }
  }

  return (
    <div className={styles.page}>
      <AssetsToolbar
        filters={filters}
        searchDraft={searchDraft}
        onSearchDraftChange={handleSearchDraftChange}
        onFiltersChange={writeFilters}
      />
      <div ref={setChromeElement} className={styles.chrome}>
        {view.inlineError ? (
          <AssetsErrorState variant="inline" onRetry={handleRetry} />
        ) : null}
        {deleteAsset.isError && !isAssetDeleteInUseError(deleteAsset.error) ? (
          <div className={styles.inlineError} role="alert">
            <div className={styles.inlineErrorCopy}>
              <Text as="span" variant="heading" size="sm">
                {ASSETS_DELETE_ERROR_TITLE}
              </Text>
              <Text as="p" variant="body" size="md">
                {ASSETS_DELETE_ERROR_DESCRIPTION}
              </Text>
            </div>
            <Button
              appearance="neutral"
              hierarchy="plain"
              size="md"
              label="Dismiss"
              onClick={() => deleteAsset.reset()}
            />
          </div>
        ) : null}
      </div>
      {body}
      {lightboxIndex >= 0 ? (
        <AssetsLightbox
          items={items}
          activeAssetId={lightboxAssetId!}
          hasNextPage={Boolean(hasNextPage) && view.inlineError === null}
          isFetchingNextPage={isFetchingNextPage}
          onFetchNextPage={fetchNextPage}
          onActiveAssetIdChange={selectLightboxAsset}
          onClose={closeLightbox}
          onDeleteAsset={(assetId) => deleteAsset.mutateAsync(assetId)}
          pendingDeleteAssetIds={deleteAsset.pendingAssetIds}
        />
      ) : null}
    </div>
  );
}
