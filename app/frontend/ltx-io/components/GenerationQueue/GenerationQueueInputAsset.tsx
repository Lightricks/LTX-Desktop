import AudioOnIcon from "@ds/assets/Icons/Audio/On.svg?react";
import VideoIcon from "@ds/assets/Icons/VideoCamera.svg?react";

import type { ExploreAsset } from "@/lib/explore-contract";
import styles from "./GenerationQueuePanel.module.scss";

export function GenerationQueueInputAsset({
  asset,
  mediaUrlForAsset,
  thumbUrlForAsset,
  isPreview,
}: {
  asset: ExploreAsset;
  mediaUrlForAsset: (asset: ExploreAsset) => string | null;
  thumbUrlForAsset: (asset: ExploreAsset) => string | null;
  isPreview?: boolean;
}) {
  const previewUrl =
    asset.media_kind === "audio"
      ? null
      : (thumbUrlForAsset(asset) ?? mediaUrlForAsset(asset));

  return (
    <div className={styles.inputAssetThumb} aria-hidden={isPreview ? true : undefined}>
      {asset.media_kind === "audio" ? (
        <span className={styles.inputAssetPlaceholder} aria-hidden>
          <AudioOnIcon />
        </span>
      ) : previewUrl ? (
        <img
          className={styles.inputAssetImage}
          src={previewUrl}
          alt={isPreview ? "" : asset.name}
        />
      ) : (
        <span className={styles.inputAssetPlaceholder} aria-hidden>
          <VideoIcon />
        </span>
      )}
    </div>
  );
}
