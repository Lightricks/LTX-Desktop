import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import type { ExploreAsset } from "@/lib/explore-contract";

import { useUploadDnd } from "../../../components/shared/Upload/hooks/useUploadDnd";
import { assetQueryKeys } from "../../../hooks/assetQueryKeys";
import { fetchAsset } from "../../../hooks/generationQueryKeys";
import { useAsset } from "../../../hooks/useAsset";
import { useExploreRuntime } from "../../../runtime/ExploreRuntime";
import { type AssetRef, type ValidationIssue, toDurableAssetRef } from "../types";

import { applyImportedAsset } from "./applyImportedAsset";

export function useMediaAssetField({
  value,
  issue,
  onChange,
  accept,
  extensions,
  chooserTitle,
  isSupportedDropped,
  isExpectedAsset,
  unsupportedMessage,
  ingestErrorMessage,
  lookupErrorMessage,
  prepareImport,
}: {
  value: AssetRef | null;
  issue?: ValidationIssue;
  onChange: (value: AssetRef | null) => void;
  accept: string;
  extensions: readonly string[];
  chooserTitle: string;
  isSupportedDropped: (file: File) => boolean;
  isExpectedAsset: (asset: ExploreAsset) => boolean;
  unsupportedMessage: string;
  ingestErrorMessage: (error: unknown) => string;
  lookupErrorMessage: (error: unknown) => string;
  /**
   * Runs between ingest and `onChange`, e.g. to trim an over-long upload.
   * `accept` may be called later or never (a cancelled trim keeps the value).
   */
  prepareImport?: (
    asset: ExploreAsset,
    accept: (asset: ExploreAsset) => void,
  ) => Promise<unknown>;
}) {
  const { api, mediaInput, mediaUrlForAsset } = useExploreRuntime();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();
  const assetQuery = useAsset(value?.assetId ?? null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [chooserBusy, setChooserBusy] = useState(false);
  // A trim modal accepts long after this render; it must not write through a
  // stale `onChange` that would drop form edits made in the meantime.
  const onChangeRef = useRef(onChange);
  useEffect(() => {
    onChangeRef.current = onChange;
  });

  const applyAsset = (asset: ExploreAsset) =>
    applyImportedAsset({
      asset,
      isExpectedAsset,
      prepareImport,
      cacheAsset: (cached) =>
        queryClient.setQueryData(assetQueryKeys.detail(cached.id), cached),
      accept: (accepted) => onChangeRef.current(toDurableAssetRef(accepted)),
      reject: () => setLocalError(unsupportedMessage),
    });

  const ingestMutation = useMutation({
    mutationFn: async (work: Promise<ExploreAsset>) => work,
    onSuccess: applyAsset,
  });

  const runIngest = (work: Promise<ExploreAsset>) => {
    setLocalError(null);
    void ingestMutation.mutateAsync(work).catch((error: unknown) => {
      setLocalError(ingestErrorMessage(error));
    });
  };

  const pickFile = (): Promise<ExploreAsset | null> => {
    setLocalError(null);
    setChooserBusy(true);
    return mediaInput
      .choose({
        title: chooserTitle,
        accept,
        extensions,
        fileInput: fileInputRef.current,
      })
      .catch((error: unknown) => {
        setLocalError(ingestErrorMessage(error));
        return null;
      })
      .finally(() => setChooserBusy(false));
  };

  const reportIngestError = (error: unknown) => {
    setLocalError(ingestErrorMessage(error));
  };

  const acceptAsset = (asset: ExploreAsset) => {
    setLocalError(null);
    void applyAsset(asset).catch(reportIngestError);
  };

  const choose = () => {
    void pickFile().then((asset) => {
      if (asset != null) acceptAsset(asset);
    });
  };

  const acceptAssetId = (assetId: string) => {
    setLocalError(null);
    void fetchAsset(api, assetId)
      .then((asset) => applyAsset(asset))
      .catch(reportIngestError);
  };

  const ingestFile = (file: File) => {
    if (!isSupportedDropped(file)) {
      setLocalError(unsupportedMessage);
      return;
    }
    runIngest(mediaInput.ingestDroppedFile(file));
  };

  const upload = useUploadDnd({
    isMultiple: false,
    acceptedFilesTypes: accept,
    onClickOverride: () => {
      choose();
    },
    onUpload: (file) => {
      ingestFile(file);
    },
  });

  const previewUrl = assetQuery.data ? mediaUrlForAsset(assetQuery.data) : null;
  const lookupError =
    value != null && assetQuery.isError ? lookupErrorMessage(assetQuery.error) : null;

  return {
    fileInputRef,
    asset: assetQuery.data ?? null,
    previewUrl,
    errorMessage: localError ?? lookupError ?? issue?.message ?? null,
    isBusy:
      chooserBusy || ingestMutation.isPending || (value != null && assetQuery.isPending),
    isDragOver: upload.isDragOver,
    dragHandlers: upload.handlers,
    choose,
    pickFile,
    acceptAsset,
    acceptAssetId,
    ingestFile,
    replace: (next: ExploreAsset) => {
      setLocalError(null);
      queryClient.setQueryData(assetQueryKeys.detail(next.id), next);
      onChangeRef.current(toDurableAssetRef(next));
    },
    clear: () => {
      setLocalError(null);
      onChange(null);
    },
    accept,
  };
}
