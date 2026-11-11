import { useAsset } from "../../../../hooks/useAsset";
import { useExploreRuntime } from "../../../../runtime/ExploreRuntime";

/** A small image URL for an asset id. Null while the asset loads or when it has no image. */
export function useAssetThumbUrl(assetId: string): string | null {
  const { mediaUrlForAsset, thumbUrlForAsset } = useExploreRuntime();
  const asset = useAsset(assetId).data;
  if (!asset) return null;
  return thumbUrlForAsset(asset) ?? mediaUrlForAsset(asset);
}
