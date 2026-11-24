import { Text } from "@ds/Text/Text";
import MediaReplaceIcon from "@ds/assets/Icons/MediaReplace.svg?react";
import PhotosIcon from "@ds/assets/Icons/Photos.svg?react";
import RemoveIcon from "@ds/assets/Icons/Remove.svg?react";
import { clsx } from "clsx";

import formStyles from "../FeatureFormView.module.scss";
import type { AssetRef, ValidationIssue } from "../types";
import { IMAGE_ACCEPT } from "./imageAssetInput";
import dropStyles from "./ImageDropzone.module.scss";
import styles from "./ImageAssetsField.module.scss";
import { useImageAssetField } from "./useImageAssetField";
import { useImageAssetsIngest } from "./useImageAssetsIngest";
import { OverlayButton } from "./OverlayButton";

function Thumb({
  label,
  value,
  onChange,
}: {
  label: string;
  value: AssetRef;
  onChange: (value: AssetRef | null) => void;
}) {
  const slot = useImageAssetField({ value, onChange, chooserTitle: label });
  return (
    <div className={styles.thumb} title={slot.errorMessage ?? undefined}>
      <input ref={slot.fileInputRef} className={dropStyles.hiddenInput} type="file" accept={slot.accept} />
      {slot.previewUrl ? <img className={styles.thumbImage} src={slot.previewUrl} alt="" /> : null}
      <div className={styles.thumbActions}>
        <OverlayButton
          label="Replace"
          ariaLabel="Replace image"
          icon={<MediaReplaceIcon />}
          onClick={slot.openLibrary}
        />
        <OverlayButton
          label="Remove"
          ariaLabel="Remove image"
          icon={<RemoveIcon />}
          onClick={slot.clear}
        />
      </div>
    </div>
  );
}

/**
 * A list of images in the ltx.io style: one large drop zone while the list is empty,
 * then a grid with an add tile first and one thumbnail for each image.
 */
export function ImageAssetsField({
  field,
  value,
  maxCount,
  issue,
  onChange,
}: {
  field: { id: string; label: string; hint?: string };
  value: AssetRef[];
  maxCount: number;
  issue?: ValidationIssue;
  onChange: (value: AssetRef[]) => void;
}) {
  const { fileInputRef, addError, clearAddError, isDragOver, dropProps } = useImageAssetsIngest({
    label: field.label,
    maxCount,
    count: value.length,
    onAdd: (added) => onChange([...value, ...added]),
  });
  const message = addError ?? issue?.message;
  return (
    <div className={formStyles.field}>
      <div className={styles.labelGroup}>
        <Text as="span" variant="body" size="md" className={formStyles.fieldLabel}>
          {field.label}
        </Text>
        {field.hint ? (
          <Text as="span" variant="body" size="sm" className={styles.hint}>
            {field.hint}
          </Text>
        ) : null}
      </div>
      <input
        ref={fileInputRef}
        className={dropStyles.hiddenInput}
        type="file"
        accept={IMAGE_ACCEPT}
        multiple
      />
      {value.length === 0 ? (
        <button
          type="button"
          className={clsx(dropStyles.dropzone, isDragOver && dropStyles.dragOver)}
          {...dropProps}
        >
          <span className={dropStyles.empty}>
            <PhotosIcon className={dropStyles.emptyIcon} aria-hidden />
            <Text as="span" variant="label" size="md">
              Add images
            </Text>
            {field.hint ? null : (
              <Text as="span" variant="body" size="md" className={styles.subtitle}>
                Upload up to {maxCount} images of characters, props, and location
              </Text>
            )}
          </span>
        </button>
      ) : (
        <div className={styles.grid}>
          {value.length < maxCount ? (
            <button
              type="button"
              className={clsx(styles.addTile, isDragOver && dropStyles.dragOver)}
              aria-label="Add image"
              {...dropProps}
            >
              +
            </button>
          ) : null}
          {value.map((image, index) => (
            <Thumb
              // Two entries can hold the same asset, so the index is the key.
              key={index}
              label={field.label}
              value={image}
              onChange={(next) => {
                // The old add message is about a list that no longer exists.
                clearAddError();
                onChange(
                  next == null
                    ? value.filter((_, at) => at !== index)
                    : value.map((current, at) => (at === index ? next : current)),
                );
              }}
            />
          ))}
        </div>
      )}
      {message ? <span className={formStyles.fieldError}>{message}</span> : null}
    </div>
  );
}
