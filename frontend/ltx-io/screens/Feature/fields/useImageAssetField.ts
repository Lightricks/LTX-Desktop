import type { AssetRef, ValidationIssue } from "../types";

import {
  IMAGE_ACCEPT,
  IMAGE_EXTENSIONS,
  imageIngestErrorMessage,
  imageLookupErrorMessage,
  isSupportedDroppedImage,
  isSupportedImageAsset,
  UNSUPPORTED_IMAGE_MESSAGE,
} from "./imageAssetInput";
import { useMediaAssetField } from "./useMediaAssetField";

/**
 * One image slot: the media field logic with the image options filled in.
 * `openLibrary` opens the asset library filtered to images the field accepts.
 */
export function useImageAssetField({
  value,
  issue,
  chooserTitle,
  onChange,
}: {
  value: AssetRef | null;
  issue?: ValidationIssue;
  chooserTitle: string;
  onChange: (value: AssetRef | null) => void;
}) {
  return useMediaAssetField({
    value,
    issue,
    onChange,
    mediaKind: "image",
    accept: IMAGE_ACCEPT,
    extensions: IMAGE_EXTENSIONS,
    chooserTitle,
    isSupportedDropped: isSupportedDroppedImage,
    isExpectedAsset: isSupportedImageAsset,
    unsupportedMessage: UNSUPPORTED_IMAGE_MESSAGE,
    ingestErrorMessage: imageIngestErrorMessage,
    lookupErrorMessage: imageLookupErrorMessage,
  });
}
