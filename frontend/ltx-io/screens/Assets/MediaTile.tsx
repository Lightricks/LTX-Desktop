import { type ReactNode, useState } from "react";

import type { ExploreListedAsset } from "@/lib/explore-contract";
import { useExploreRuntime } from "@/ltx-io/runtime/ExploreRuntime";

import { AssetImageTile } from "./AssetImageTile";
import { AssetTileActions } from "./AssetTileActions";
import { AssetVideoTile } from "./AssetVideoTile";
import { AssetsAudioTile } from "./AssetsAudioTile";
import styles from "./MediaTile.module.scss";
import { shouldOpenAssetFromKey } from "./lightboxCollection";
import { isNestedInteractiveTarget, useFinePointer } from "./tileInteraction";

export function MediaTile({
  asset,
  onOpen,
  onDelete,
  isDeletePending = false,
  labelVerb = "Open",
}: {
  asset: ExploreListedAsset;
  onOpen: () => void;
  onDelete?: (assetId: string) => Promise<void>;
  isDeletePending?: boolean;
  labelVerb?: "Open" | "Select";
}) {
  const runtime = useExploreRuntime();
  const isFinePointer = useFinePointer();
  const [isHovered, setIsHovered] = useState(false);
  const mediaUrl = runtime.mediaUrlForAsset(asset);
  const thumbnailUrl = runtime.thumbUrlForAsset(asset);
  const shouldHoverPlay = runtime.hoverPreview && isFinePointer;

  let media: ReactNode;
  switch (asset.media_kind) {
    case "image":
      media = (
        <AssetImageTile asset={asset} mediaUrl={mediaUrl} thumbnailUrl={thumbnailUrl} />
      );
      break;
    case "video":
      media = (
        <AssetVideoTile
          asset={asset}
          mediaUrl={mediaUrl}
          thumbnailUrl={thumbnailUrl}
          shouldHoverPlay={shouldHoverPlay}
          isHovered={isHovered}
        />
      );
      break;
    case "audio":
      media = (
        <AssetsAudioTile
          asset={asset}
          isHovered={isHovered}
          shouldHoverPlay={shouldHoverPlay}
        />
      );
      break;
    default: {
      const _exhaustive: never = asset.media_kind;
      media = _exhaustive;
    }
  }

  return (
    <div
      className={styles.tile}
      role="button"
      tabIndex={0}
      aria-label={`${labelVerb} ${asset.name}`}
      onClick={onOpen}
      onKeyDown={(event) => {
        if (
          shouldOpenAssetFromKey(
            event.key,
            isNestedInteractiveTarget(event.target, event.currentTarget)
              ? "nested"
              : "tile",
          )
        ) {
          event.preventDefault();
          onOpen();
        }
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {media}
      {onDelete ? (
        <AssetTileActions
          asset={asset}
          mediaUrl={mediaUrl}
          onDelete={onDelete}
          isDeletePending={isDeletePending}
        />
      ) : null}
    </div>
  );
}
