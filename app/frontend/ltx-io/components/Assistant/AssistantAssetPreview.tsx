import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { Tooltip } from "@ds/Tooltip/Tooltip";
import { RotateCcw } from "lucide-react";
import { useState } from "react";

import { useAssetDragSource } from "@/lib/asset-drag";
import { VideoPlayer } from "@/ltx-io/components/VideoView/VideoPlayer";
import { useAsset } from "@/ltx-io/hooks/useAsset";
import { useExploreRuntime } from "@/ltx-io/runtime/ExploreRuntime";
import { AssetsLightbox } from "@/ltx-io/screens/Assets/AssetsLightbox";

import styles from "./AssistantAssetPreview.module.scss";
import { assistantMedia } from "./assistant-media";

export function AssistantAssetPreview({ assetId }: { assetId: string }) {
  const query = useAsset(assetId);
  const runtime = useExploreRuntime();
  const [mediaFailed, setMediaFailed] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(false);
  const asset = query.data;
  const { Icon } = assistantMedia[asset?.media_kind ?? "unknown"];
  const url = asset ? runtime.mediaUrlForAsset(asset) : null;
  const dragSource = useAssetDragSource(asset?.id ?? null);

  return (
    <figure className={styles.asset} aria-busy={query.isPending}>
      {query.isPending ? (
        <div className={styles.placeholder}>Loading asset...</div>
      ) : query.isError ? (
        <div className={styles.placeholder}>
          Asset unavailable.
          <Tooltip content="Retry loading asset">
            <Button
              appearance="neutral"
              hierarchy="plain"
              size="sm"
              isIconOnly
              leftIcon={<RotateCcw size={16} />}
              aria-label="Retry loading asset"
              onClick={() => void query.refetch()}
            />
          </Tooltip>
        </div>
      ) : asset && url ? (
        <>
          <div
            className={styles.media}
            onErrorCapture={() => setMediaFailed(true)}
            {...dragSource}
          >
            {asset.media_kind === "video" ? (
              <VideoPlayer
                src={url}
                posterUrl={runtime.thumbUrlForAsset(asset) ?? undefined}
                label={asset.name}
              />
            ) : asset.media_kind === "image" ? (
              <button
                type="button"
                className={styles.imageButton}
                aria-label={`Preview ${asset.name}`}
                onClick={() => setPreviewOpen(true)}
              >
                <img src={url} alt={asset.name} loading="lazy" />
              </button>
            ) : (
              <audio src={url} controls preload="metadata" aria-label={asset.name} />
            )}
          </div>
          {mediaFailed && <small role="status">Could not load this media file.</small>}
          {previewOpen && (
            <AssetsLightbox
              items={[{ ...asset, in_use: true }]}
              activeAssetId={asset.id}
              onClose={() => setPreviewOpen(false)}
            />
          )}
          <figcaption>
            <Icon size={14} />
            <Tooltip content={asset.name}>
              <Text as="span" variant="body" size="sm" shouldTruncate title={asset.name}>
                {asset.name}
              </Text>
            </Tooltip>
          </figcaption>
        </>
      ) : (
        <div className={styles.placeholder}>Asset unavailable.</div>
      )}
    </figure>
  );
}
