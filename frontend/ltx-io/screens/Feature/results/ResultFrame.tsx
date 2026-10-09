import { type ReactNode } from "react";

import { Button } from "@ds/Button/Button";
import { Flex } from "@ds/layout/Flex/Flex";
import { Tooltip } from "@ds/Tooltip/Tooltip";
import FailLineIcon from "@ds/assets/Icons/Fail/Line.svg?react";
import RemoveIcon from "@ds/assets/Icons/Remove.svg?react";
import RetryIcon from "@ds/assets/Icons/Reset.svg?react";
import CloseIcon from "@ds/assets/Icons/Close/Normal.svg?react";

import { DotsLoader } from "../../../components/shared/DotsLoader/DotsLoader";
import { LoadingPulse } from "../../../components/shared/LoadingPulse/LoadingPulse";
import { VideoPlayer } from "../../../components/VideoView/VideoPlayer";
import { useAssetDownload } from "../../../hooks/useAssetDownload";
import { promptEmbeddingFailureNotice } from "../../../lib/promptEmbeddingFailure";
import { rejectedLtxApiKeyNotice } from "../../../lib/rejectedLtxApiKey";
import { useExploreRuntime } from "../../../runtime/ExploreRuntime";
import { isInFlightGeneration, type Generation } from "../../../lib/resultsFeedModel";
import { CutoutResult } from "./cutout/CutoutResult";
import { pickPlayback } from "./cutout/cutoutOutput";
import { inFlightResultStatus } from "./inFlightResultStatus";
import { REMOVE_FROM_HISTORY_LABEL } from "./resultActionsCopy";
import { resolveResultFileAction } from "./resultFileAction";
import { ResultFileActionButton } from "./ResultFileActionButton";
import { ResultMetaBar } from "./ResultMetaBar";
import { ResultStatus } from "./ResultStatus";
import styles from "./ResultFrame.module.scss";

function InFlightProgress({
  status,
  progressPercent,
}: {
  status: string;
  progressPercent: number | undefined;
}) {
  const presentation = inFlightResultStatus({ status, progressPercent });
  // A percent is plain text. A live region would announce every poll tick.
  const announce = presentation.showDots || status === "cancelling";

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
  /** The feature returns a cutout WebM, which plays over a backdrop. */
  isCutoutFeature?: boolean;
};

export function ResultFrame({
  generation,
  onCancel,
  onRetry,
  onDelete,
  progressPercent,
  isCutoutFeature = false,
}: ResultFrameProps) {
  const runtime = useExploreRuntime();
  const downloadResultFile = useAssetDownload();
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
  const failureNotice = rejectedKey ?? embeddingFailed;
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

  return (
    <Flex direction="column" gap="sm">
      <ResultMetaBar generation={generation} />
      <div className={styles.outputStackItem}>
        {isInFlightGeneration(generation) ? (
          <>
            <LoadingPulse />
            <InFlightProgress
              status={generation.status}
              progressPercent={progressPercent}
            />
            <div className={styles.overlayActions}>
              {generation.status !== "cancelling" ? (
                <OverlayButton
                  label="Cancel"
                  icon={<CloseIcon />}
                  onClick={() => onCancel(generation.id)}
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
                onClick={() => onDelete(generation.id)}
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
                onClick={() => onDelete(generation.id)}
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
                onClick={() => onDelete(generation.id)}
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
