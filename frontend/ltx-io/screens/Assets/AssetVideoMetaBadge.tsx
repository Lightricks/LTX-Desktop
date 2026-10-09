import type { ExploreListedAsset } from "@/lib/explore-contract";

import { formatVideoMeta } from "./assetVideoMeta";
import styles from "./MediaTile.module.scss";

const SEPARATOR = " \u00b7 ";

/** Read-only chip. It ignores pointer events, so clicks reach the tile. */
export function AssetVideoMetaBadge({
  metadata,
}: {
  metadata: ExploreListedAsset["metadata"];
}) {
  const parts = formatVideoMeta(metadata);
  if (parts.length === 0) return null;

  return <div className={styles.meta}>{parts.join(SEPARATOR)}</div>;
}
