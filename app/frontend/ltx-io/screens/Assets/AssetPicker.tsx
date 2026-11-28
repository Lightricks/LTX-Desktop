import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { useInfiniteQuery } from "@tanstack/react-query";
import { clsx } from "clsx";
import { type DragEvent, type ReactNode, useCallback, useEffect, useState } from "react";

import type { ExploreAsset } from "@/lib/explore-contract";
import { useModalContext } from "@/ltx-io/components/shared/Modal/ModalContext";

import { assetQueryKeys } from "../../hooks/assetQueryKeys";
import { unwrapApiResult } from "../../lib/unwrapApiResult";
import { useExploreRuntime } from "../../runtime/ExploreRuntime";

import styles from "./AssetPicker.module.scss";
import { AssetsErrorState } from "./AssetsErrorState";
import { AssetsGrid } from "./AssetsGrid";
import { resolveAssetPickerDrop } from "./assetPickerDrop";
import { resolveAssetsLibraryView } from "./assetsLibraryView";
import { type AssetsMediaKind } from "./assetsSearchParams";

const PICKER_MODAL_NAME = "asset-picker";

function pickerTitle(mediaKind: AssetsMediaKind | null): string {
  switch (mediaKind) {
    case null:
      return "Assets";
    case "video":
      return "Videos";
    case "audio":
      return "Audio";
    case "image":
      return "Images";
    default: {
      const _exhaustive: never = mediaKind;
      return _exhaustive;
    }
  }
}

function emptyTitle(mediaKind: AssetsMediaKind | null): string {
  switch (mediaKind) {
    case null:
      return "No assets yet";
    case "video":
      return "No videos yet";
    case "audio":
      return "No audio yet";
    case "image":
      return "No images yet";
    default: {
      const _exhaustive: never = mediaKind;
      return _exhaustive;
    }
  }
}

function useAssetsByKind(mediaKind: AssetsMediaKind | null) {
  const { api } = useExploreRuntime();
  const listFilters = {
    media_kind: mediaKind ?? undefined,
    sort: "created_at-desc" as const,
  };

  return useInfiniteQuery({
    queryKey: assetQueryKeys.list(listFilters),
    initialPageParam: null as string | null,
    queryFn: async ({ pageParam }) =>
      unwrapApiResult(
        await api.listAssets({
          ...listFilters,
          cursor: pageParam,
        }),
      ),
    getNextPageParam: (last) => last.next_cursor ?? undefined,
  });
}

export function AssetPicker({
  mediaKind,
  isEligibleAsset,
  onSelect,
  onDropData,
  onUpload,
  onUploadMany,
}: {
  /** null lists every kind. */
  mediaKind: AssetsMediaKind | null;
  /**
   * Hides assets the field cannot take, for example a GIF cutout in an image field.
   * It keeps the pages as the API returns them, so a page of hidden assets shows
   * fewer tiles. While no tile shows and more pages exist, the picker loads the next
   * page. If short pages are a problem, filter by MIME type in `listAssets`.
   */
  isEligibleAsset?: (asset: ExploreAsset) => boolean;
  onSelect: (asset: ExploreAsset) => void;
  onDropData: (dataTransfer: DataTransfer) => void;
  /** Device file chooser. Resolves null when the user cancels. */
  onUpload?: () => Promise<ExploreAsset | null>;
  /**
   * Device chooser for several files. It replaces `onUpload` and adds the files itself.
   * Resolves true when the user chose files, false when the user cancels.
   */
  onUploadMany?: () => Promise<boolean>;
}) {
  const query = useAssetsByKind(mediaKind);
  const [scrollRoot, setScrollRoot] = useState<HTMLDivElement | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  // Resolves true when the user chose files. `onUploadMany` handles its files itself.
  const upload =
    onUploadMany ??
    (onUpload
      ? async () => {
          const asset = await onUpload();
          if (asset != null) onSelect(asset);
          return asset != null;
        }
      : undefined);
  const items = (query.data?.pages.flatMap((page) => page.items) ?? []).filter(
    (asset) => isEligibleAsset?.(asset) ?? true,
  );
  const { hasNextPage, isFetchingNextPage, isFetchNextPageError, fetchNextPage } = query;
  const shownCount = items.length;
  // A loaded page can be all hidden. The grid has no tile to scroll, so it would not load more.
  useEffect(() => {
    if (shownCount === 0 && hasNextPage && !isFetchingNextPage && !isFetchNextPageError) {
      void fetchNextPage();
    }
  }, [shownCount, hasNextPage, isFetchingNextPage, isFetchNextPageError, fetchNextPage]);
  const view = resolveAssetsLibraryView({
    isLoading: query.isLoading,
    error: query.error,
    loadedItemCount: items.length,
    isFetchNextPageError: query.isFetchNextPageError,
    hasNextPage: Boolean(query.hasNextPage),
  });

  const handleRetry = () => {
    if (view.retryTarget === "fetchNextPage") {
      void query.fetchNextPage();
      return;
    }
    void query.refetch();
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
          <Text as="p" variant="body" size="md">
            {emptyTitle(mediaKind)}
          </Text>
        </div>
      );
      break;
    case "loading":
    case "content":
      body = (
        <AssetsGrid
          items={items}
          isLoading={query.isLoading && items.length === 0}
          hasNextPage={Boolean(query.hasNextPage) && view.inlineError === null}
          isFetchingNextPage={query.isFetchingNextPage}
          layoutRoot={scrollRoot}
          scrollRoot={scrollRoot}
          labelVerb="Select"
          onNextPage={() => {
            void query.fetchNextPage();
          }}
          onOpenAsset={onSelect}
        />
      );
      break;
    default: {
      const _exhaustive: never = view.kind;
      body = _exhaustive;
    }
  }

  const allowDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  };

  return (
    <div
      className={clsx(styles.body, isDragOver && styles.dragOver)}
      onDragEnter={(event) => {
        allowDrop(event);
        setIsDragOver(true);
      }}
      onDragOver={allowDrop}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node)) {
          setIsDragOver(false);
        }
      }}
      onDrop={(event) => {
        event.preventDefault();
        event.stopPropagation();
        setIsDragOver(false);
        onDropData(event.dataTransfer);
      }}
    >
      <div className={styles.header}>
        <Text as="h2" variant="heading" size="sm">
          {pickerTitle(mediaKind)}
        </Text>
        {upload ? (
          <Button
            appearance="neutral"
            hierarchy="plain"
            size="md"
            label="Upload from device"
            disabled={isUploading}
            onClick={() => {
              // Start in this click so a phone can open its file picker.
              setIsUploading(true);
              void upload()
                .then((chosen) => {
                  if (!chosen) setIsUploading(false);
                })
                .catch(() => setIsUploading(false));
            }}
          />
        ) : null}
      </div>
      {view.inlineError ? (
        <AssetsErrorState variant="inline" onRetry={handleRetry} />
      ) : null}
      <div ref={setScrollRoot} className={styles.scroll}>
        {body}
      </div>
    </div>
  );
}

export function useOpenAssetPicker({
  mediaKind,
  isEligibleAsset,
  isEligibleFile,
  onSelect,
  onDropFile,
  onDropAssetId,
  onUpload,
  onUploadMany,
}: {
  /** null lists every kind. */
  mediaKind: AssetsMediaKind | null;
  /** Hides listed assets the field cannot take. */
  isEligibleAsset?: (asset: ExploreAsset) => boolean;
  isEligibleFile: (file: File) => boolean;
  onSelect: (asset: ExploreAsset) => void;
  onDropFile: (file: File) => void;
  onDropAssetId: (assetId: string) => void;
  onUpload?: () => Promise<ExploreAsset | null>;
  /** Several files from the device. The picker closes when the user chooses files. */
  onUploadMany?: () => Promise<boolean>;
}) {
  const { showModal, hideModal } = useModalContext();

  return useCallback(() => {
    const take = (apply: () => void) => {
      // Close first. A clip over the duration cap opens the trim modal in
      // the same turn, and that call replaces this one.
      hideModal("user_selection");
      apply();
    };

    showModal({
      content: (
        <AssetPicker
          mediaKind={mediaKind}
          isEligibleAsset={isEligibleAsset}
          onSelect={(asset) => take(() => onSelect(asset))}
          onUpload={onUpload}
          onUploadMany={
            onUploadMany &&
            (async () => {
              const chosen = await onUploadMany();
              if (chosen) hideModal("user_selection");
              return chosen;
            })
          }
          onDropData={(dataTransfer) => {
            const decision = resolveAssetPickerDrop({
              assetPayload: dataTransfer.getData("asset"),
              file: dataTransfer.files[0] ?? null,
              mediaKind,
              isEligibleFile,
            });
            switch (decision.kind) {
              case "asset":
                take(() => onDropAssetId(decision.id));
                return;
              case "file":
                take(() => onDropFile(decision.file));
                return;
              case "ignore":
                return;
              default: {
                const _exhaustive: never = decision;
                return _exhaustive;
              }
            }
          }}
        />
      ),
      modalName: PICKER_MODAL_NAME,
      modalClassName: styles.frame,
      variant: "primary",
      noPadding: true,
    });
  }, [
    hideModal,
    isEligibleAsset,
    isEligibleFile,
    mediaKind,
    onDropAssetId,
    onDropFile,
    onSelect,
    onUpload,
    onUploadMany,
    showModal,
  ]);
}
