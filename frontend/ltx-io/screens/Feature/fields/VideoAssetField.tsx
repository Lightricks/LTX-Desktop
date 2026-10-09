import { ActivityCircular } from "@ds/ActivityCircular/ActivityCircular";
import { Text } from "@ds/Text/Text";
import VideoIcon from "@ds/assets/Icons/Video.svg?react";
import { clsx } from "clsx";

import type { ExploreAsset } from "@/lib/explore-contract";

import { isVideoAsset } from "../../../hooks/assetQueryKeys";
import { unwrapApiResult } from "../../../lib/unwrapApiResult";
import { useExploreRuntime } from "../../../runtime/ExploreRuntime";
import { useOpenAssetPicker } from "../../Assets/AssetPicker";
import formStyles from "../FeatureFormView.module.scss";
import { VIDEO_DURATION_CAP_TOLERANCE_SECONDS } from "../trim/trimConstants";
import { useTrimVideoModal } from "../trim/useTrimModal";
import type { AssetRef, ValidationIssue } from "../types";

import styles from "./VideoAssetField.module.scss";
import { VideoAssetPreview, type VideoAssetRangeValue } from "./VideoAssetPreview";
import {
  assetDurationSeconds,
  prepareDurationLimitedMediaImport,
} from "./prepareDurationLimitedMediaImport";
import { useMediaAssetField } from "./useMediaAssetField";
import {
  UNSUPPORTED_VIDEO_MESSAGE,
  VIDEO_ACCEPT,
  VIDEO_EXTENSIONS,
  isSupportedDroppedVideo,
  videoIngestErrorMessage,
  videoLookupErrorMessage,
} from "./videoAssetInput";

export function VideoAssetField({
  field,
  value,
  issue,
  onChange,
  maxDurationSeconds,
  range,
}: {
  field: { id: string; label: string };
  value: AssetRef | null;
  issue?: ValidationIssue;
  onChange: (value: AssetRef | null) => void;
  /** Longer uploads open the trim modal before they are accepted. */
  maxDurationSeconds?: number;
  range?: VideoAssetRangeValue;
}) {
  const { api } = useExploreRuntime();
  const openTrimModal = useTrimVideoModal();
  const prepareImport =
    maxDurationSeconds == null
      ? undefined
      : (asset: ExploreAsset, accept: (asset: ExploreAsset) => void) =>
          prepareDurationLimitedMediaImport({
            asset,
            trimCapSeconds: maxDurationSeconds,
            toleranceSeconds: VIDEO_DURATION_CAP_TOLERANCE_SECONDS,
            trim: async (assetId, startSec, endSec) =>
              unwrapApiResult(await api.trimVideo(assetId, { startSec, endSec })),
            onAccept: accept,
            openTrim: openTrimModal,
          });

  const media = useMediaAssetField({
    value,
    issue,
    onChange,
    accept: VIDEO_ACCEPT,
    extensions: VIDEO_EXTENSIONS,
    chooserTitle: field.label,
    isSupportedDropped: isSupportedDroppedVideo,
    isExpectedAsset: isVideoAsset,
    unsupportedMessage: UNSUPPORTED_VIDEO_MESSAGE,
    ingestErrorMessage: videoIngestErrorMessage,
    lookupErrorMessage: videoLookupErrorMessage,
    prepareImport,
  });
  const openLibrary = useOpenAssetPicker({
    mediaKind: "video",
    isEligibleFile: isSupportedDroppedVideo,
    onSelect: media.acceptAsset,
    onUpload: media.pickFile,
    onDropFile: media.ingestFile,
    onDropAssetId: media.acceptAssetId,
  });
  const durationSeconds = media.asset ? assetDurationSeconds(media.asset) : null;
  const openSourceTrim = () => {
    const asset = media.asset;
    if (asset == null || durationSeconds == null || durationSeconds <= 0) return;
    openTrimModal({
      asset,
      durationSeconds,
      capSeconds:
        maxDurationSeconds == null
          ? durationSeconds
          : Math.min(maxDurationSeconds, durationSeconds),
      subtitle: "Select the part of the video to keep.",
      onSave: async (startSec, endSec) => {
        media.replace(
          unwrapApiResult(await api.trimVideo(asset.id, { startSec, endSec })),
        );
      },
    });
  };

  return (
    <div className={formStyles.field}>
      <Text as="span" variant="body" size="md" className={formStyles.fieldLabel}>
        {field.label}
      </Text>
      <input
        ref={media.fileInputRef}
        className={styles.hiddenInput}
        type="file"
        accept={media.accept}
      />
      {media.previewUrl ? (
        <VideoAssetPreview
          src={media.previewUrl}
          isBusy={media.isBusy}
          isDragOver={media.isDragOver}
          durationSeconds={durationSeconds}
          range={range}
          onReplace={openLibrary}
          onTrim={
            durationSeconds != null && durationSeconds > 0 ? openSourceTrim : undefined
          }
          onRemove={media.clear}
          dragHandlers={media.dragHandlers}
        />
      ) : (
        <button
          type="button"
          className={clsx(styles.dropzone, media.isDragOver && styles.dragOver)}
          onClick={openLibrary}
          onDragEnter={media.dragHandlers.handleDragEnter}
          onDragOver={media.dragHandlers.handleDragOver}
          onDragLeave={media.dragHandlers.handleDragLeave}
          onDrop={media.dragHandlers.handleDrop}
          aria-label={field.label}
        >
          <div className={styles.empty}>
            <VideoIcon className={styles.emptyIcon} />
            <Text as="span" variant="body" size="md" className={styles.emptyTitle}>
              Add video
            </Text>
          </div>
          {media.isBusy ? (
            <div className={styles.overlay}>
              <ActivityCircular size={28} />
            </div>
          ) : null}
        </button>
      )}
      {media.errorMessage ? (
        <span className={formStyles.fieldError}>{media.errorMessage}</span>
      ) : null}
    </div>
  );
}
