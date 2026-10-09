import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { useInfiniteQuery } from "@tanstack/react-query";
import { clsx } from "clsx";
import { type DragEvent, type ReactNode, useCallback, useState } from "react";

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

function pickerTitle(mediaKind: AssetsMediaKind): string {
  switch (mediaKind) {
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

function emptyTitle(mediaKind: AssetsMediaKind): string {
  switch (mediaKind) {
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

// TODO: when a screen opens this picker with `mediaKind="image"`, filter the items by the
// types the field accepts (`IMAGE_MIME_TYPES` in `imageAssetInput.ts`). A cutout run stores
// a GIF as an image asset, and no image field accepts it. Add an `isEligibleAsset` prop and
// leave the Assets library screen unfiltered, so it still shows the GIF. A filter here
// keeps the pages as the API returns them. If short pages are a problem, add a MIME type
// filter to `listAssets` instead.
function useAssetsByKind(mediaKind: AssetsMediaKind) {
  const { api } = useExploreRuntime();
  const listFilters = { media_kind: mediaKind, sort: "created_at-desc" as const };

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
  onSelect,
  onDropData,
  onUpload,
}: {
  mediaKind: AssetsMediaKind;
  onSelect: (asset: ExploreAsset) => void;
  onDropData: (dataTransfer: DataTransfer) => void;
  /** Device file chooser. Resolves null when the user cancels. */
  onUpload?: () => Promise<ExploreAsset | null>;
}) {
  const query = useAssetsByKind(mediaKind);
  const [scrollRoot, setScrollRoot] = useState<HTMLDivElement | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const items = query.data?.pages.flatMap((page) => page.items) ?? [];
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
        {onUpload ? (
          <Button
            appearance="neutral"
            hierarchy="plain"
            size="md"
            label="Upload from device"
            disabled={isUploading}
            onClick={() => {
              // Start in this click so a phone can open its file picker.
              setIsUploading(true);
              void onUpload()
                .then((asset) => {
                  if (asset != null) onSelect(asset);
                  else setIsUploading(false);
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
  isEligibleFile,
  onSelect,
  onDropFile,
  onDropAssetId,
  onUpload,
}: {
  mediaKind: AssetsMediaKind;
  isEligibleFile: (file: File) => boolean;
  onSelect: (asset: ExploreAsset) => void;
  onDropFile: (file: File) => void;
  onDropAssetId: (assetId: string) => void;
  onUpload?: () => Promise<ExploreAsset | null>;
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
          onSelect={(asset) => take(() => onSelect(asset))}
          onUpload={onUpload}
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
    isEligibleFile,
    mediaKind,
    onDropAssetId,
    onDropFile,
    onSelect,
    onUpload,
    showModal,
  ]);
}
