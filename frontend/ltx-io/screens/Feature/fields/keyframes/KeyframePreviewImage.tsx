import { useAsset } from "../../../../hooks/useAsset";
import { useExploreRuntime } from "../../../../runtime/ExploreRuntime";

/** The full image of a still. It shows nothing while the asset loads or when it has no image. */
export function KeyframePreviewImage({
  assetId,
  className,
}: {
  assetId: string;
  className?: string;
}) {
  const { mediaUrlForAsset } = useExploreRuntime();
  const asset = useAsset(assetId).data;
  const url = asset ? mediaUrlForAsset(asset) : null;
  return url ? <img className={className} src={url} alt="" draggable={false} /> : null;
}
