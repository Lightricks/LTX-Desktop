import { Button } from "@ds/Button/Button";
import { useTheme } from "@ds/styles/themes/useTheme";
import * as Dialog from "@radix-ui/react-dialog";
import { ChevronLeft, ChevronRight, FolderOpen, X } from "lucide-react";
import {
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import type { ExploreListedAsset } from "@/lib/explore-contract";
import { revealInFolderLabel } from "@/lib/revealInFolderLabel";
import { useAssetDownload } from "@/ltx-io/hooks/useAssetDownload";
import { useExploreRuntime } from "@/ltx-io/runtime/ExploreRuntime";
import { isAlphaWebmMime } from "@/ltx-io/screens/Feature/results/cutout/cutoutOutput";

import { AssetDeleteButton, OverlayButton } from "./AssetTileActions";
import styles from "./AssetsLightbox.module.scss";
import { AssetsLightboxAudio } from "./AssetsLightboxAudio";
import { AssetsLightboxVideo } from "./AssetsLightboxVideo";
import { resolveAssetLibraryFileAction } from "./assetLibraryFileAction";
import {
  type LightboxArrowKeyTarget,
  findLightboxAssetIndex,
  lightboxMediaKey,
  nextLightboxIndex,
  resolveLightboxIndex,
  shouldHandleLightboxArrowKey,
} from "./lightboxCollection";
import {
  isLightboxGallerySwipePointer,
  isLightboxSwipeIgnoredTarget,
  swipeGalleryDirection,
} from "./lightboxSwipe";

const BLOCKED_ARROW_TARGET_SELECTOR = [
  "audio",
  "input",
  "select",
  "textarea",
  "video",
  "[contenteditable]",
  "[role='slider']",
].join(",");

function lightboxArrowKeyTarget(target: EventTarget | null): LightboxArrowKeyTarget {
  if (!(target instanceof Element)) return "content";
  if (target.closest("[data-lightbox-chrome]") !== null) return "chrome";
  if (target instanceof HTMLElement && target.isContentEditable) return "blocked";
  return target.closest(BLOCKED_ARROW_TARGET_SELECTOR) === null ? "content" : "blocked";
}

function LightboxMedia({
  asset,
  src,
}: {
  asset: ExploreListedAsset;
  src: string | null;
}) {
  const [hasError, setHasError] = useState(false);
  const handleError = useCallback(() => setHasError(true), []);

  if (asset.media_kind === "audio") {
    return <AssetsLightboxAudio asset={asset} />;
  }

  if (src === null || hasError) {
    return <div className={styles.brokenMedia}>Media unavailable</div>;
  }

  switch (asset.media_kind) {
    case "image":
      return (
        <img className={styles.media} src={src} alt={asset.name} onError={handleError} />
      );
    case "video":
      return (
        <AssetsLightboxVideo
          src={src}
          isCutout={isAlphaWebmMime(asset.mime_type)}
          onError={handleError}
        />
      );
    default: {
      const _exhaustive: never = asset.media_kind;
      return _exhaustive;
    }
  }
}

export function AssetsLightbox({
  items,
  activeAssetId,
  activeIndex,
  hasNextPage = false,
  isFetchingNextPage = false,
  onFetchNextPage = async () => undefined,
  onActiveAssetIdChange = () => undefined,
  onActiveIndexChange,
  onClose,
  onDeleteAsset,
  pendingDeleteAssetIds,
  allowReveal = true,
}: {
  items: ExploreListedAsset[];
  activeAssetId: string;
  /**
   * The position of the active item. Pass it when `items` can hold one asset more than once, because
   * an id then matches the first copy only. Without it, the id picks the item.
   */
  activeIndex?: number;
  hasNextPage?: boolean;
  isFetchingNextPage?: boolean;
  onFetchNextPage?: () => Promise<unknown>;
  onActiveAssetIdChange?: (assetId: string) => void;
  /** Called with the new position when the user steps. Use it with `activeIndex`. */
  onActiveIndexChange?: (index: number) => void;
  onClose: () => void;
  onDeleteAsset?: (assetId: string) => Promise<void>;
  pendingDeleteAssetIds?: ReadonlySet<string>;
  /** Set to false for a preview of an input, where the button to open the file in the file manager has no use. */
  allowReveal?: boolean;
}) {
  const runtime = useExploreRuntime();
  const download = useAssetDownload();
  const { rootElement } = useTheme();
  const swipeStartRef = useRef<{ x: number; y: number } | null>(null);
  const suppressClickRef = useRef(false);
  const [pendingNextFromAssetId, setPendingNextFromAssetId] = useState<string | null>(
    null,
  );
  const index = resolveLightboxIndex(items, activeAssetId, activeIndex);
  const asset = index >= 0 ? items[index] : undefined;
  const gallerySize = items.length;
  const mediaUrl = asset ? runtime.mediaUrlForAsset(asset) : null;
  const canGoPrevious = index > 0;
  const nextAction = nextLightboxIndex(
    index,
    gallerySize,
    hasNextPage,
    isFetchingNextPage,
  );
  const canGoNext = nextAction !== "end";
  const showNavigation = gallerySize > 1;

  useEffect(() => {
    if (index < 0) {
      onClose();
      return;
    }

    if (pendingNextFromAssetId === null) return;

    const pendingIndex = findLightboxAssetIndex(items, pendingNextFromAssetId);
    if (pendingIndex < 0) {
      setPendingNextFromAssetId(null);
      return;
    }

    const nextItem = items[pendingIndex + 1];
    if (activeAssetId === pendingNextFromAssetId && nextItem) {
      onActiveAssetIdChange(nextItem.id);
      setPendingNextFromAssetId(null);
      return;
    }

    if (!hasNextPage && !isFetchingNextPage) {
      setPendingNextFromAssetId(null);
    }
  }, [
    activeAssetId,
    hasNextPage,
    index,
    isFetchingNextPage,
    items,
    onActiveAssetIdChange,
    onClose,
    pendingNextFromAssetId,
  ]);

  const goPrevious = useCallback(() => {
    const previousItem = items[index - 1];
    if (!previousItem) return;
    setPendingNextFromAssetId(null);
    onActiveAssetIdChange(previousItem.id);
    onActiveIndexChange?.(index - 1);
  }, [index, items, onActiveAssetIdChange, onActiveIndexChange]);

  const goNext = useCallback(() => {
    switch (nextAction) {
      case "next": {
        const nextItem = items[index + 1];
        if (nextItem) {
          onActiveAssetIdChange(nextItem.id);
          onActiveIndexChange?.(index + 1);
        }
        return;
      }
      case "fetch":
        setPendingNextFromAssetId(activeAssetId);
        void onFetchNextPage();
        return;
      case "end":
        return;
      default: {
        const _exhaustive: never = nextAction;
        return _exhaustive;
      }
    }
  }, [
    activeAssetId,
    index,
    items,
    nextAction,
    onActiveAssetIdChange,
    onActiveIndexChange,
    onFetchNextPage,
  ]);

  const handleSwipePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (!isLightboxGallerySwipePointer(event.pointerType)) return;
      if (isLightboxSwipeIgnoredTarget(event.target)) {
        swipeStartRef.current = null;
        return;
      }
      swipeStartRef.current = { x: event.clientX, y: event.clientY };
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [],
  );

  const handleSwipePointerUp = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const start = swipeStartRef.current;
      swipeStartRef.current = null;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
      if (!start) return;
      const direction = swipeGalleryDirection(
        start.x,
        start.y,
        event.clientX,
        event.clientY,
      );
      if (direction === null) return;
      suppressClickRef.current = true;
      if (direction === "previous") goPrevious();
      else goNext();
    },
    [goNext, goPrevious],
  );

  const handleSwipePointerCancel = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      swipeStartRef.current = null;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
    },
    [],
  );

  if (!asset) return null;

  const fileAction = resolveAssetLibraryFileAction({
    revealInFolder: runtime.revealInFolder,
    path: asset.path,
    mediaUrl,
  });

  return (
    <Dialog.Root open onOpenChange={(open) => !open && onClose()}>
      <Dialog.Portal container={rootElement ?? undefined}>
        <Dialog.Overlay
          className={styles.overlay}
          onClick={(event) => {
            if (event.target === event.currentTarget) onClose();
          }}
        >
          <Dialog.Content
            className={styles.content}
            onPointerDownOutside={(event) => event.preventDefault()}
            onInteractOutside={(event) => event.preventDefault()}
            onPointerDown={handleSwipePointerDown}
            onPointerUp={handleSwipePointerUp}
            onPointerCancel={handleSwipePointerCancel}
            onClickCapture={(event) => {
              if (!suppressClickRef.current) return;
              suppressClickRef.current = false;
              event.preventDefault();
              event.stopPropagation();
            }}
            onKeyDown={(event) => {
              const canNavigate = event.key === "ArrowLeft" ? canGoPrevious : canGoNext;
              if (
                !shouldHandleLightboxArrowKey(
                  event.key,
                  lightboxArrowKeyTarget(event.target),
                  canNavigate,
                )
              ) {
                return;
              }
              event.preventDefault();
              event.stopPropagation();
              if (event.key === "ArrowLeft") goPrevious();
              else goNext();
            }}
          >
            <Dialog.Title className={styles.screenReaderOnly}>{asset.name}</Dialog.Title>
            <Dialog.Description className={styles.screenReaderOnly}>
              Fullscreen preview of {asset.name}
            </Dialog.Description>
            <div className={styles.header} data-lightbox-chrome>
              {runtime.deleteAsset !== null && onDeleteAsset ? (
                <AssetDeleteButton
                  asset={asset}
                  onDelete={onDeleteAsset}
                  isDeletePending={pendingDeleteAssetIds?.has(asset.id) ?? false}
                />
              ) : null}
              {allowReveal && fileAction.kind === "reveal" ? (
                <OverlayButton
                  label={revealInFolderLabel()}
                  icon={<FolderOpen />}
                  onClick={() => runtime.revealInFolder?.(fileAction.path)}
                />
              ) : null}
              <Dialog.Close asChild>
                <Button
                  appearance="overlay"
                  hierarchy="secondary"
                  size="md"
                  isIconOnly
                  leftIcon={<X />}
                  aria-label="Close"
                  data-lightbox-chrome
                />
              </Dialog.Close>
            </div>
            <div className={styles.mediaStage}>
              <LightboxMedia
                key={lightboxMediaKey(asset.id, mediaUrl)}
                asset={asset}
                src={mediaUrl}
              />
            </div>{" "}
            {fileAction.kind === "download" ? (
              <div className={styles.footer}>
                <button
                  type="button"
                  className={styles.download}
                  onClick={() => download(asset)}
                  data-lightbox-chrome
                >
                  Download
                </button>
              </div>
            ) : null}
            {showNavigation ? (
              <>
                <Button
                  appearance="overlay"
                  hierarchy="secondary"
                  size="lg"
                  isIconOnly
                  leftIcon={<ChevronLeft />}
                  aria-label="Previous"
                  className={styles.previous}
                  disabled={!canGoPrevious}
                  onClick={goPrevious}
                  data-lightbox-chrome
                />
                <Button
                  appearance="overlay"
                  hierarchy="secondary"
                  size="lg"
                  isIconOnly
                  leftIcon={<ChevronRight />}
                  aria-label="Next"
                  className={styles.next}
                  disabled={!canGoNext}
                  onClick={goNext}
                  data-lightbox-chrome
                />
              </>
            ) : null}
          </Dialog.Content>
        </Dialog.Overlay>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
