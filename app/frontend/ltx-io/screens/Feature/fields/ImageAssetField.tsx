import { ActivityCircular } from "@ds/ActivityCircular/ActivityCircular";
import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import PhotoIcon from "@ds/assets/Icons/Photo.svg?react";
import { clsx } from "clsx";
import type { ReactNode } from "react";

import type { ExploreAsset } from "@/lib/explore-contract";

import formStyles from "../FeatureFormView.module.scss";
import type { AssetRef, ValidationIssue } from "../types";
import dropStyles from "./ImageDropzone.module.scss";
import styles from "./ImageAssetField.module.scss";
import { useImageAssetField } from "./useImageAssetField";

export function ImageAssetField({
  field,
  value,
  naturalPreview = false,
  issue,
  action,
  onChange,
}: {
  field: { id: string; label: string };
  /** Beside the label: a way to make the image instead of uploading it. */
  action?: (input: { setAsset: (asset: ExploreAsset) => void }) => ReactNode;
  /** Show the whole image at its own aspect ratio, not in a fixed-height box. */
  naturalPreview?: boolean;
  value: AssetRef | null;
  issue?: ValidationIssue;
  onChange: (value: AssetRef | null) => void;
}) {
  const media = useImageAssetField({
    value,
    issue,
    onChange,
    chooserTitle: field.label,
  });

  return (
    <div className={formStyles.field}>
      <div className={styles.labelRow}>
        <Text as="span" variant="body" size="md" className={formStyles.fieldLabel}>
          {field.label}
        </Text>
        {action?.({ setAsset: media.acceptAsset })}
      </div>
      {/* Outside the dropzone: the Remote chooser clicks it, and that click must not reopen the library. */}
      <input
        ref={media.fileInputRef}
        className={dropStyles.hiddenInput}
        type="file"
        accept={media.accept}
      />
      <div
        className={clsx(dropStyles.dropzone, media.isDragOver && dropStyles.dragOver)}
        onDragEnter={media.dragHandlers.handleDragEnter}
        onDragOver={media.dragHandlers.handleDragOver}
        onDragLeave={media.dragHandlers.handleDragLeave}
        onDrop={media.dragHandlers.handleDrop}
        onClick={media.openLibrary}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            media.openLibrary();
          }
        }}
        aria-label={field.label}
      >
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
          <div className={dropStyles.empty}>
            <PhotoIcon className={dropStyles.emptyIcon} />
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
            media.openLibrary();
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
