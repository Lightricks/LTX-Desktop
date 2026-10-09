import { ActivityCircular } from "@ds/ActivityCircular/ActivityCircular";
import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import PhotoIcon from "@ds/assets/Icons/Photo.svg?react";
import { clsx } from "clsx";

import { isImageAsset } from "../../../hooks/assetQueryKeys";
import formStyles from "../FeatureFormView.module.scss";
import type { AssetRef, ValidationIssue } from "../types";
import {
  IMAGE_ACCEPT,
  IMAGE_EXTENSIONS,
  imageIngestErrorMessage,
  imageLookupErrorMessage,
  isSupportedDroppedImage,
  UNSUPPORTED_IMAGE_MESSAGE,
} from "./imageAssetInput";
import styles from "./ImageAssetField.module.scss";
import { useMediaAssetField } from "./useMediaAssetField";

export function ImageAssetField({
  field,
  value,
  naturalPreview = false,
  issue,
  onChange,
}: {
  field: { id: string; label: string };
  /** Show the whole image at its own aspect ratio, not in a fixed-height box. */
  naturalPreview?: boolean;
  value: AssetRef | null;
  issue?: ValidationIssue;
  onChange: (value: AssetRef | null) => void;
}) {
  const media = useMediaAssetField({
    value,
    issue,
    onChange,
    accept: IMAGE_ACCEPT,
    extensions: IMAGE_EXTENSIONS,
    chooserTitle: field.label,
    isSupportedDropped: isSupportedDroppedImage,
    isExpectedAsset: isImageAsset,
    unsupportedMessage: UNSUPPORTED_IMAGE_MESSAGE,
    ingestErrorMessage: imageIngestErrorMessage,
    lookupErrorMessage: imageLookupErrorMessage,
  });

  return (
    <div className={formStyles.field}>
      <Text as="span" variant="body" size="md" className={formStyles.fieldLabel}>
        {field.label}
      </Text>
      <div
        className={clsx(styles.dropzone, media.isDragOver && styles.dragOver)}
        onDragEnter={media.dragHandlers.handleDragEnter}
        onDragOver={media.dragHandlers.handleDragOver}
        onDragLeave={media.dragHandlers.handleDragLeave}
        onDrop={media.dragHandlers.handleDrop}
        onClick={media.dragHandlers.handleInputClick}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            media.dragHandlers.handleInputClick();
          }
        }}
        aria-label={field.label}
      >
        <input
          ref={media.fileInputRef}
          className={styles.hiddenInput}
          type="file"
          accept={media.accept}
        />
        {media.previewUrl ? (
          <img
            className={clsx(
              styles.preview,
              naturalPreview && styles.previewNatural,
            )}
            src={media.previewUrl}
            alt=""
          />
        ) : (
          <div className={styles.empty}>
            <PhotoIcon className={styles.emptyIcon} />
            <Text as="span" variant="body" size="md">
              Drop an image or browse
            </Text>
          </div>
        )}
        {media.isBusy ? (
          <div className={styles.overlay}>
            <ActivityCircular size={28} />
          </div>
        ) : null}
      </div>
      <div className={styles.actions}>
        <Button
          appearance="neutral"
          hierarchy="secondary"
          size="md"
          label={value ? "Replace" : "Browse"}
          onClick={(event) => {
            event.stopPropagation();
            media.choose();
          }}
        />
        {value ? (
          <Button
            appearance="neutral"
            hierarchy="secondary"
            size="md"
            label="Clear"
            onClick={(event) => {
              event.stopPropagation();
              media.clear();
            }}
          />
        ) : null}
      </div>
      {media.errorMessage ? (
        <span className={formStyles.fieldError}>{media.errorMessage}</span>
      ) : null}
    </div>
  );
}
