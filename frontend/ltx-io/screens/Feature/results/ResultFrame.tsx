import { type ReactNode } from "react";

import { Button } from "@ds/Button/Button";
import { Flex } from "@ds/layout/Flex/Flex";
import { Tooltip } from "@ds/Tooltip/Tooltip";
import FailLineIcon from "@ds/assets/Icons/Fail/Line.svg?react";
import RemoveIcon from "@ds/assets/Icons/Remove.svg?react";
import RetryIcon from "@ds/assets/Icons/Reset.svg?react";
import CloseIcon from "@ds/assets/Icons/Close/Normal.svg?react";
import { useAssetDragSource } from "@/lib/asset-drag";

import { DotsLoader } from "../../../components/shared/DotsLoader/DotsLoader";
import { LoadingPulse } from "../../../components/shared/LoadingPulse/LoadingPulse";
import { VideoPlayer } from "../../../components/VideoView/VideoPlayer";
import { useAssetDownload } from "../../../hooks/useAssetDownload";
import { GENERATION_CANCEL_CONFIRMATION } from "../../../hooks/useCancelGeneration";
import { useConfirmDeleteAction } from "../../../hooks/useConfirmDeleteAction";
import { promptEmbeddingFailureNotice } from "../../../lib/promptEmbeddingFailure";
import { staticFailureNotice } from "../../../lib/staticFailureNotices";
import { rejectedLtxApiKeyNotice } from "../../../lib/rejectedLtxApiKey";
import { useExploreRuntime } from "../../../runtime/ExploreRuntime";
import { isInFlightGeneration, type Generation } from "../../../lib/resultsFeedModel";
import { CutoutResult } from "./cutout/CutoutResult";
import { pickPlayback } from "./cutout/cutoutOutput";
import { inFlightResultStatus } from "./inFlightResultStatus";
import {
  REMOVE_FROM_HISTORY_LABEL,
  resultDeleteConfirmation,
} from "./resultActionsCopy";
import { resolveResultFileAction } from "./resultFileAction";
import { ResultFileActionButton } from "./ResultFileActionButton";
import { ResultMetaBar } from "./ResultMetaBar";
import { ResultStatus } from "./ResultStatus";
import styles from "./ResultFrame.module.scss";

function InFlightProgress({
  status,
  progressPercent,
  remainingSeconds,
}: {
  status: string;
  progressPercent: number | undefined;
  remainingSeconds?: number | null;
}) {
  const presentation = inFlightResultStatus({
    status,
    progressPercent,
    remainingSeconds,
  });
  // A percent or countdown is plain text. A live region would announce every poll tick.
  const announce =
    (presentation.showDots && presentation.body == null) ||
    status === "cancelling";

  return (
    <div
      className={styles.generationProgress}
      data-testid="generation-progress"
      role={announce ? "status" : undefined}
      aria-label={presentation.showDots ? "Generating" : undefined}
    >
      <ResultStatus
        icon={presentation.showDots ? <DotsLoader /> : undefined}
        title={presentation.title}
        body={presentation.body}
        bodyClassName={styles.inFlightBody}
      />
    </div>
  );
}

function OverlayButton({
  label,
  icon,
  onClick,
}: {
  label: string;
  icon: ReactNode;
  onClick: () => void;
}) {
  return (
    <Tooltip content={label}>
      <span>
        <Button
          appearance="overlay"
          hierarchy="secondary"
          size="md"
          isIconOnly
          leftIcon={icon}
          aria-label={label}
          onClick={onClick}
        />
      </span>
    </Tooltip>
  );
}

export type ResultFrameProps = {
  generation: Generation;
  onCancel: (id: string) => void;
  onRetry: (id: string) => void;
  onDelete: (id: string) => void;
  /** Determinate 0–95 progress for this generation, when the queue has reported it. */
  progressPercent?: number;
  /** Live remaining seconds for this generation, when the queue has reported it. */
  remainingSeconds?: number;
  /** The feature returns a cutout WebM, which plays over a backdrop. */
  isCutoutFeature?: boolean;
};

export function ResultFrame({
  generation,
  onCancel,
  onRetry,
  onDelete,
  progressPercent,
  remainingSeconds,
  isCutoutFeature = false,
}: ResultFrameProps) {
  const runtime = useExploreRuntime();
  const confirmDelete = useConfirmDeleteAction();
  const downloadResultFile = useAssetDownload();
  const handleCancel = () =>
    confirmDelete(GENERATION_CANCEL_CONFIRMATION, () => onCancel(generation.id));
  const hasAsset =
    generation.status !== "failed" && generation.status !== "cancelled";
  const handleDelete = () =>
    confirmDelete(resultDeleteConfirmation(hasAsset), () =>
      onDelete(generation.id),
    );
  const rejectedKey = rejectedLtxApiKeyNotice({
    errorCode: generation.error_code,
    usesLtxApiTextEncoding: runtime.usesLtxApiTextEncoding,
    canOpenSettings: runtime.openLtxApiKeySettings != null,
  });
  const embeddingFailed = promptEmbeddingFailureNotice({
    errorCode: generation.error_code,
    usesLtxApiTextEncoding: runtime.usesLtxApiTextEncoding,
    canOpenSettings: runtime.openTextEncodingSettings != null,
  });
  const staticFailure = staticFailureNotice({
    errorCode: generation.error_code,
    ranOnBet: generation.risk != null,
  });
  const failureNotice = rejectedKey ?? embeddingFailed ?? staticFailure;
  const openFailureSettings = rejectedKey
    ? runtime.openLtxApiKeySettings
    : runtime.openTextEncodingSettings;
  const { output, asCutout } = pickPlayback(
    generation,
    navigator.userAgent,
    isCutoutFeature,
  );
  const mediaUrl = output ? runtime.mediaUrlForAsset(output) : null;
  const fileAction =
    generation.status === "succeeded"
      ? resolveResultFileAction({
          generation,
          playedOutput: output,
          isCutoutFeature,
          revealInFolder: runtime.revealInFolder,
          mediaUrlForAsset: runtime.mediaUrlForAsset,
          download: downloadResultFile,
        })
      : null;
  const dragSource = useAssetDragSource(
    generation.status === "succeeded" && output ? output.id : null,
  );

  return (
    <Flex direction="column" gap="sm">
      <ResultMetaBar generation={generation} />
      <div className={styles.outputStackItem} {...dragSource}>
        {isInFlightGeneration(generation) ? (
          <>
            <LoadingPulse />
            <InFlightProgress
              status={generation.status}
              progressPercent={progressPercent}
              remainingSeconds={remainingSeconds}
            />
            <div className={styles.overlayActions}>
              {generation.status !== "cancelling" ? (
                <OverlayButton
                  label="Cancel"
                  icon={<CloseIcon />}
                  onClick={handleCancel}
                />
              ) : null}
            </div>
          </>
        ) : generation.status === "failed" ? (
          <>
            <ResultStatus
              icon={<FailLineIcon />}
              title="Generation failed"
              body={failureNotice?.body ?? generation.error_code ?? undefined}
              copyClassName={failureNotice ? styles.rejectedKeyCopy : undefined}
              action={
                failureNotice?.openSettingsLabel && openFailureSettings ? (
                  <Button
                    appearance="neutral"
                    hierarchy="primary"
                    size="lg"
                    label={failureNotice.openSettingsLabel}
                    onClick={openFailureSettings}
                  />
                ) : undefined
              }
            />
            <div className={styles.overlayActions}>
              <OverlayButton
                label="Retry"
                icon={<RetryIcon />}
                onClick={() => onRetry(generation.id)}
              />
              <OverlayButton
                label={REMOVE_FROM_HISTORY_LABEL}
                icon={<RemoveIcon />}
                onClick={handleDelete}
              />
            </div>
          </>
        ) : generation.status === "cancelled" ? (
          <>
            <ResultStatus title="Cancelled" />
            <div className={styles.overlayActions}>
              <OverlayButton
                label={REMOVE_FROM_HISTORY_LABEL}
                icon={<RemoveIcon />}
                onClick={handleDelete}
              />
            </div>
          </>
        ) : mediaUrl ? (
          <>
            {asCutout ? (
              <CutoutResult src={mediaUrl} />
            ) : (
              <VideoPlayer src={mediaUrl} />
            )}
            <div className={styles.overlayActions}>
              <OverlayButton
                label={REMOVE_FROM_HISTORY_LABEL}
                icon={<RemoveIcon />}
                onClick={handleDelete}
              />
              {fileAction ? <ResultFileActionButton action={fileAction} /> : null}
            </div>
          </>
        ) : (
          <ResultStatus
            icon={<FailLineIcon />}
            title="Generation failed"
            body="OUTPUT_MISSING"
          />
        )}
      </div>
    </Flex>
  );
}
