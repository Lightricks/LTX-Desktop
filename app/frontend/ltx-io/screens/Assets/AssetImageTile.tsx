import type { ExploreListedAsset } from "@/lib/explore-contract";

import { AssetMetaBadge } from "./AssetMetaBadge";
import { VideoPlaceholder } from "./AssetVideoTile";
import styles from "./MediaTile.module.scss";

export function AssetImageTile({
  asset,
  mediaUrl,
  thumbnailUrl,
}: {
  asset: ExploreListedAsset;
  mediaUrl: string | null;
  thumbnailUrl: string | null;
}) {
  const imageUrl = thumbnailUrl ?? mediaUrl;

  return imageUrl ? (
    <>
      <img
        className={styles.image}
        src={imageUrl}
        alt={asset.name}
        draggable={false}
      />
      <AssetMetaBadge metadata={asset.metadata} />
    </>
  ) : (
    <VideoPlaceholder kind="missing" />
  );
}
