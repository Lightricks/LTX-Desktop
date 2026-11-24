import { ActivityCircular } from "@ds/ActivityCircular/ActivityCircular";
import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import AudioIcon from "@ds/assets/Icons/Audio/On.svg?react";
import RemoveIcon from "@ds/assets/Icons/Remove.svg?react";
import { useQueryClient } from "@tanstack/react-query";
import { clsx } from "clsx";
import { useCallback, useMemo, useRef, useState } from "react";

import type { ExploreAsset } from "@/lib/explore-contract";
import { AudioWaveformPlayer } from "@/ltx-io/components/AudioPlayer/AudioWaveformPlayer";
import { assetQueryKeys, isAudioAsset } from "../../../hooks/assetQueryKeys";
import { useAsset } from "../../../hooks/useAsset";
import { useUploadDnd } from "../../../components/shared/Upload/hooks/useUploadDnd";
import { unwrapApiResult } from "../../../lib/unwrapApiResult";
import { useExploreRuntime } from "../../../runtime/ExploreRuntime";
import { useAudioSourceUrl } from "../trim/useAudioSourceUrl";
import { useTrimAudioModal } from "../trim/useTrimModal.tsx";
import formStyles from "../FeatureFormView.module.scss";
import {
  toDurableAssetRef,
  type AssetRef,
  type ValidationIssue,
} from "../types";
import {
  AUDIO_ACCEPT,
  AUDIO_EXTENSIONS,
  AUDIO_OR_VIDEO_ACCEPT,
  AUDIO_OR_VIDEO_EXTENSIONS,
  audioIngestErrorMessage,
  audioLookupErrorMessage,
  isSupportedDroppedAudio,
  UNSUPPORTED_AUDIO_MESSAGE,
  VIDEO_AUDIO_UNSUPPORTED_MESSAGE,
} from "./audioAssetInput";
import { isSupportedDroppedVideo } from "./videoAssetInput";
import { AudioSourceMenu, buildAudioSourceMenuItems } from "./audioSourceMenu";
import { recordedAudioToWavFile } from "@/ltx-io/media/microphone/encodePcmWav";
import { prepareDurationLimitedMediaImport } from "./prepareDurationLimitedMediaImport";
import { useAudioRecorderModal } from "./useAudioRecorderModal";
import styles from "./AudioAssetField.module.scss";

const FALLBACK_RECORD_SECONDS = 20;

export function AudioAssetField({
  field,
  value,
  issue,
  onChange,
  trimCapSeconds,
  trimMinSeconds,
}: {
  field: { id: string; label: string };
  value: AssetRef | null;
  issue?: ValidationIssue;
  onChange: (value: AssetRef | null) => void;
  trimCapSeconds: number | null;
  /** Shortest selection the trim modal accepts. Defaults to the Explore floor. */
  trimMinSeconds?: number;
}) {
  const { api, mediaInput, ingestRecordedFile, ingestDroppedVideoAudio, microphoneAccess } =
    useExploreRuntime();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();
  const assetQuery = useAsset(value?.assetId ?? null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const openRecorder = useAudioRecorderModal();
  const canExtractVideo = ingestDroppedVideoAudio != null;
  const accept = canExtractVideo ? AUDIO_OR_VIDEO_ACCEPT : AUDIO_ACCEPT;
  const extensions = canExtractVideo
    ? AUDIO_OR_VIDEO_EXTENSIONS
    : AUDIO_EXTENSIONS;

  const cacheAsset = useCallback(
    (asset: ExploreAsset) => {
      queryClient.setQueryData(assetQueryKeys.detail(asset.id), asset);
    },
    [queryClient],
  );

  const acceptAsset = useCallback(
    (asset: ExploreAsset) => {
      cacheAsset(asset);
      if (!isAudioAsset(asset)) {
        setLocalError(UNSUPPORTED_AUDIO_MESSAGE);
        return;
      }
      onChange(toDurableAssetRef(asset));
    },
    [cacheAsset, onChange],
  );

  const trimAudio = useCallback(
    async (assetId: string, startSec: number, endSec: number) =>
      unwrapApiResult(await api.trimAudio(assetId, { startSec, endSec })),
    [api],
  );

  const openTrimModal = useTrimAudioModal({
    minDurationSeconds: trimMinSeconds,
  });

  const runImport = useCallback(
    (work: Promise<ExploreAsset | null>) => {
      setLocalError(null);
      setIsBusy(true);
      void work
        .then(async (asset) => {
          if (asset == null) return;
          cacheAsset(asset);
          await prepareDurationLimitedMediaImport({
            asset,
            trimCapSeconds,
            trim: trimAudio,
            onAccept: acceptAsset,
            openTrim: openTrimModal,
          });
        })
        .catch((error: unknown) => {
          setLocalError(audioIngestErrorMessage(error));
        })
        .finally(() => setIsBusy(false));
    },
    [cacheAsset, trimCapSeconds, trimAudio, acceptAsset, openTrimModal],
  );

  const chooseAudio = useCallback(() => {
    runImport(
      mediaInput.choose({
        title: field.label,
        accept,
        extensions,
        fileInput: fileInputRef.current,
      }),
    );
  }, [accept, extensions, field.label, mediaInput, runImport]);

  const ingestRecording = useCallback(
    (file: File) => {
      const wav =
        file.type === "audio/wav"
          ? Promise.resolve(file)
          : recordedAudioToWavFile(file);
      runImport(wav.then((ready) => ingestRecordedFile(ready)));
    },
    [ingestRecordedFile, runImport],
  );

  const canRecord = microphoneAccess != null;

  const recordAudio = useCallback(() => {
    if (microphoneAccess == null) {
      return;
    }
    const maxDurationSeconds = trimCapSeconds ?? FALLBACK_RECORD_SECONDS;
    openRecorder({
      access: microphoneAccess,
      maxDurationSeconds,
      onUseAudio: ingestRecording,
    });
  }, [ingestRecording, microphoneAccess, openRecorder, trimCapSeconds]);

  const sourceMenuItems = useMemo(
    () =>
      buildAudioSourceMenuItems({
        onImport: chooseAudio,
        onRecord: canRecord ? recordAudio : undefined,
        importLabel: "Import audio file",
      }),
    [canRecord, chooseAudio, recordAudio],
  );

  const upload = useUploadDnd({
    isMultiple: false,
    acceptedFilesTypes: accept,
    onUpload: (file) => {
      if (isSupportedDroppedAudio(file)) {
        runImport(mediaInput.ingestDroppedFile(file));
        return;
      }
      if (isSupportedDroppedVideo(file)) {
        const extract = ingestDroppedVideoAudio;
        if (extract == null) {
          setLocalError(VIDEO_AUDIO_UNSUPPORTED_MESSAGE);
          return;
        }
        runImport(extract(file));
        return;
      }
      setLocalError(UNSUPPORTED_AUDIO_MESSAGE);
    },
  });

  const asset = assetQuery.data ?? null;
  const source = useAudioSourceUrl(asset);
  const lookupError =
    value != null && assetQuery.isError
      ? audioLookupErrorMessage(assetQuery.error)
      : null;
  const notice =
    localError == null && lookupError == null && issue?.blocksGenerate === false
      ? issue.message
      : null;
  const errorMessage = notice
    ? null
    : (localError ?? lookupError ?? issue?.message ?? null);
  const showSpinner = isBusy || (value != null && assetQuery.isPending);
  const dragClass = upload.isDragOver ? styles.dragOver : undefined;

  return (
    <div className={formStyles.field}>
      <Text
        as="span"
        variant="body"
        size="md"
        className={formStyles.fieldLabel}
      >
        {field.label}
      </Text>
      {/* Mounted once per field so Remote choose can drive a real picker. */}
      <input
        ref={fileInputRef}
        className={styles.hiddenInput}
        type="file"
        accept={accept}
        tabIndex={-1}
        aria-hidden="true"
      />
      {asset ? (
        <div
          className={clsx(styles.card, dragClass)}
          onDragEnter={upload.handlers.handleDragEnter}
          onDragOver={upload.handlers.handleDragOver}
          onDragLeave={upload.handlers.handleDragLeave}
          onDrop={upload.handlers.handleDrop}
        >
          <div className={styles.preview}>
            <AudioWaveformPlayer
              src={source.status === "ready" ? source.url : ""}
              disabled={source.status !== "ready"}
            />
          </div>
          <div className={styles.footer}>
            <AudioSourceMenu enabled items={sourceMenuItems}>
              <Button
                appearance="neutral"
                hierarchy="plain"
                size="md"
                label="Replace"
                className={styles.replaceAction}
                aria-label="Replace audio"
              />
            </AudioSourceMenu>
            <Button
              appearance="neutral"
              hierarchy="plain"
              size="md"
              label="Remove"
              leftIcon={<RemoveIcon />}
              className={styles.replaceAction}
              aria-label="Remove audio"
              onClick={(event) => {
                event.stopPropagation();
                setLocalError(null);
                onChange(null);
              }}
            />
          </div>
          {showSpinner ? (
            <div className={styles.overlay}>
              <ActivityCircular size={28} />
            </div>
          ) : null}
        </div>
      ) : (
        <AudioSourceMenu enabled items={sourceMenuItems}>
          <button
            type="button"
            className={clsx(styles.dropzone, dragClass)}
            onDragEnter={upload.handlers.handleDragEnter}
            onDragOver={upload.handlers.handleDragOver}
            onDragLeave={upload.handlers.handleDragLeave}
            onDrop={upload.handlers.handleDrop}
            aria-label={field.label}
          >
            <div className={styles.empty}>
              <AudioIcon className={styles.emptyIcon} />
              <Text
                as="span"
                variant="body"
                size="md"
                className={styles.emptyTitle}
              >
                Add audio
              </Text>
              <Text as="span" variant="body" size="sm">
                {canRecord ? "Upload or record" : "Import audio file"}
              </Text>
            </div>
            {showSpinner ? (
              <div className={styles.overlay}>
                <ActivityCircular size={28} />
              </div>
            ) : null}
          </button>
        </AudioSourceMenu>
      )}
      {notice ? (
        <span className={formStyles.fieldNotice}>{notice}</span>
      ) : errorMessage ? (
        <span className={formStyles.fieldError}>{errorMessage}</span>
      ) : null}
    </div>
  );
}
