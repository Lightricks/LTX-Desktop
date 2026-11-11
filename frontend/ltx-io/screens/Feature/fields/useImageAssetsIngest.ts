import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { useUploadDnd } from "../../../components/shared/Upload/hooks/useUploadDnd";
import { assetQueryKeys } from "../../../hooks/assetQueryKeys";
import { fetchAsset } from "../../../hooks/generationQueryKeys";
import { type PendingImport, useExploreRuntime } from "../../../runtime/ExploreRuntime";
import { useOpenAssetPicker } from "../../Assets/AssetPicker";
import { type AssetRef, toDurableAssetRef } from "../types";
import {
  IMAGE_ACCEPT,
  IMAGE_EXTENSIONS,
  imageIngestErrorMessage,
  isSupportedDroppedImage,
  isSupportedImageAsset,
} from "./imageAssetInput";
import { collectImageBatch, describeImageBatch, planImageBatch } from "./imageBatch";

/**
 * Add many images to a list field: drop files, pick from the library, or upload from the device.
 * The field keeps its own list. This hook only reports the images it accepted, in one `onAdd`
 * call for each batch, and says how many it can still take through `count` and `maxCount`.
 */
export function useImageAssetsIngest<Placement = never>({
  label,
  maxCount,
  count,
  onAdd,
}: {
  label: string;
  maxCount: number;
  /** How many images the list holds now. */
  count: number;
  /** `placement` is the value given to `dropAt`. It is empty for any other way to add. */
  onAdd: (assets: AssetRef[], placement?: Placement) => void;
}) {
  const { api, mediaInput } = useExploreRuntime();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [addError, setAddError] = useState<string | null>(null);
  // How many batches are being imported now. A batch is not on the list until it ends.
  const [importing, setImporting] = useState(0);
  // An ingest ends after this render. It must add to the list as it is then.
  const latest = useRef({ count, onAdd });
  useEffect(() => {
    latest.current = { count, onAdd };
  });

  // Add a whole batch with one `onAdd`. Separate adds would each start from the same old list.
  const addImports = async (
    pending: PendingImport[],
    unsupported: number,
    placement?: Placement,
  ) => {
    setAddError(null);
    setImporting((count) => count + 1);
    try {
      await importBatch(pending, unsupported, placement);
    } finally {
      setImporting((count) => count - 1);
    }
  };

  const importBatch = async (
    pending: PendingImport[],
    unsupported: number,
    placement?: Placement,
  ) => {
    const plan = planImageBatch(pending, maxCount - latest.current.count);
    const results = await Promise.allSettled(plan.take.map((run) => run()));
    const batch = collectImageBatch(results, isSupportedImageAsset);
    for (const asset of batch.assets) {
      queryClient.setQueryData(assetQueryKeys.detail(asset.id), asset);
    }
    // Another add can finish first, so count the free slots again now.
    const { count: current, onAdd: add } = latest.current;
    const accepted = batch.assets.slice(0, Math.max(0, maxCount - current));
    if (accepted.length > 0) {
      add(accepted.map(toDurableAssetRef), placement);
    }
    setAddError(
      describeImageBatch({
        unsupported: unsupported + batch.wrongKind,
        overLimit: plan.overLimit + (batch.assets.length - accepted.length),
        maxCount,
        failures: batch.failures,
      }),
    );
  };

  // Set by `dropAt` only while the drop handler runs. `onUpload` runs inside it, before it returns.
  const dropPlacement = useRef<Placement | undefined>(undefined);

  const dropFiles = (files: File[]) => {
    const supported = files.filter(isSupportedDroppedImage);
    void addImports(
      supported.map((file) => () => mediaInput.ingestDroppedFile(file)),
      files.length - supported.length,
      dropPlacement.current,
    );
  };

  // A library pick adds one image. "Upload from device" and drops add many files.
  const openLibrary = useOpenAssetPicker({
    mediaKind: "image",
    isEligibleAsset: isSupportedImageAsset,
    isEligibleFile: isSupportedDroppedImage,
    onSelect: (asset) => void addImports([() => Promise.resolve(asset)], 0),
    onUploadMany: () => {
      setAddError(null);
      return mediaInput
        .chooseMany({
          title: label,
          accept: IMAGE_ACCEPT,
          extensions: IMAGE_EXTENSIONS,
          fileInput: fileInputRef.current,
        })
        .then((pending) => {
          void addImports(pending, 0);
          return pending.length > 0;
        })
        .catch((error: unknown) => {
          setAddError(imageIngestErrorMessage(error));
          return false;
        });
    },
    onDropFile: (file) => dropFiles([file]),
    onDropAssetId: (assetId) => void addImports([() => fetchAsset(api, assetId)], 0),
  });

  const upload = useUploadDnd({
    isMultiple: true,
    acceptedFilesTypes: IMAGE_ACCEPT,
    onUpload: dropFiles,
  });
  const dropProps = {
    onDragEnter: upload.handlers.handleDragEnter,
    onDragOver: upload.handlers.handleDragOver,
    onDragLeave: upload.handlers.handleDragLeave,
    onDrop: upload.handlers.handleDrop,
    onClick: openLibrary,
  };

  /** Drop files and tell `onAdd` where they landed, for example a point on a timeline. */
  const dropAt = (event: React.DragEvent<HTMLElement>, placement: Placement) => {
    dropPlacement.current = placement;
    try {
      upload.handlers.handleDrop(event);
    } finally {
      dropPlacement.current = undefined;
    }
  };

  return {
    /** The hidden `<input type="file" multiple>` the device upload clicks. */
    fileInputRef,
    addError,
    clearAddError: () => setAddError(null),
    isDragOver: upload.isDragOver,
    /** True while a batch of images is imported. Those images are not on the list yet. */
    isImporting: importing > 0,
    dropProps,
    dropAt,
    openLibrary,
  };
}
