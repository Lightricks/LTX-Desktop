import { Text } from "@ds/Text/Text";
import { type ReactNode, useMemo, useState } from "react";

import type { ExploreAsset } from "@/lib/explore-contract";

import formStyles from "../FeatureFormView.module.scss";
import { MotionPathEditModal } from "../motionPath/MotionPathEditModal";
import { MotionPathRefTile } from "../motionPath/MotionPathRefTile";
import { useMotionTrackFrameAspect } from "../motionPath/useMotionTrackFrameAspect";
import type { MotionPath } from "../../../lib/motionPathSampler";
import type { AssetRef, FieldOption, MotionPathFieldEdit, ValidationIssue } from "../types";
import dropStyles from "./ImageDropzone.module.scss";
import imageFieldStyles from "./ImageAssetField.module.scss";
import { useImageAssetField } from "./useImageAssetField";

/**
 * The reference image of a Motion Track run with the path drawn on it, as on ltx.io.
 * The tile shows the image and the path. A click opens the two-step path editor.
 * A new image opens the editor at once. The definition clears the old path.
 */
export function MotionPathField({
  field,
  image,
  paths,
  duration,
  issue,
  action,
  onChange,
}: {
  field: { id: string; label: string; durationOptions: FieldOption<number>[] };
  /** Beside the label: a way to make the image instead of uploading it. */
  action?: (input: { setAsset: (asset: ExploreAsset) => void }) => ReactNode;
  image: AssetRef | null;
  paths: MotionPath[];
  duration: number;
  issue?: ValidationIssue;
  onChange: (edit: MotionPathFieldEdit) => void;
}) {
  const [isEditing, setIsEditing] = useState(false);
  const media = useImageAssetField({
    value: image,
    issue,
    onChange: (nextImage) => {
      onChange({ image: nextImage });
      setIsEditing(nextImage != null);
    },
    chooserTitle: field.label,
  });
  const { previewUrl } = media;
  // The one place that loads the image to read its frame. The tile and the editor get the result.
  const imageAspect = useMotionTrackFrameAspect(previewUrl ?? undefined);
  const tracks = useMemo(() => paths.map((path) => path.points), [paths]);

  return (
    <div className={formStyles.field}>
      <div className={imageFieldStyles.labelRow}>
        <Text as="span" variant="body" size="md" className={formStyles.fieldLabel}>
          {field.label}
        </Text>
        {action?.({ setAsset: media.acceptAsset })}
      </div>
      <input
        ref={media.fileInputRef}
        className={dropStyles.hiddenInput}
        type="file"
        accept={media.accept}
      />
      <div
        onDragEnter={media.dragHandlers.handleDragEnter}
        onDragOver={media.dragHandlers.handleDragOver}
        onDragLeave={media.dragHandlers.handleDragLeave}
        onDrop={media.dragHandlers.handleDrop}
      >
        <MotionPathRefTile
          tracks={tracks}
          imageUrl={previewUrl}
          imageAspect={imageAspect}
          onEdit={() => setIsEditing(true)}
          onUpload={media.openLibrary}
          onReplace={previewUrl ? media.openLibrary : undefined}
        />
      </div>
      {media.errorMessage ? (
        <span className={formStyles.fieldError}>{media.errorMessage}</span>
      ) : null}
      {isEditing && previewUrl ? (
        <MotionPathEditModal
          imageUrl={previewUrl}
          imageAspect={imageAspect}
          paths={paths}
          durationSec={duration}
          durationOptions={field.durationOptions
            .filter((option) => option.disabledReason == null)
            .map((option) => option.value)}
          onAccept={(nextPaths, durationSec) => {
            onChange({ paths: nextPaths, duration: durationSec });
            setIsEditing(false);
          }}
          onClose={() => setIsEditing(false)}
        />
      ) : null}
    </div>
  );
}
