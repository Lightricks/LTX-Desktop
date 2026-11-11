import { ActivityCircular } from "@ds/ActivityCircular/ActivityCircular";
import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { clsx } from "clsx";
import { type CSSProperties, useEffect, useState } from "react";

import { EnhanceSplitButton } from "@/components/EnhanceSplitButton";
import { enhanceActionLabel } from "@/lib/enhance-ui-copy";
import type { ExploreAsset } from "@/lib/explore-contract";
import {
  DEFAULT_IMAGE_RESOLUTION,
  type ImageResolution,
} from "@/lib/genspace-image-resolution";
import {
  type PromptHistoryEntry,
  resolvePromptProvenance,
} from "@/lib/prompt-provenance";

import {
  GENERATION_CANCEL_CONFIRMATION,
  useCancelGeneration,
} from "../../hooks/useCancelGeneration";
import { useConfirmDeleteAction } from "../../hooks/useConfirmDeleteAction";
import { useImageEnhance } from "../../hooks/useImageEnhance";
import { isInFlightGeneration } from "../../lib/resultsFeedModel";
import { useExploreRuntime } from "../../runtime/ExploreRuntime";

import styles from "./ImageGenerator.module.scss";
import { ImageProviderPicker } from "./ImageProviderPicker";
import { ImageSizeFields } from "./ImageSizeFields";
import { imageFailureMessage } from "./imageFailureMessage";
import {
  type ImageAspect,
  aspectRatioNumber,
  buildTextToImageRequest,
  fittingImageResolution,
  imageSizeForAspect,
} from "./imageGeneratorRequest";
import { imageCreateErrorMessage } from "./imageProviderAvailability";
import { useImageProviderChoice } from "./useImageProviderChoice";
import { useOwnedGeneration } from "./useOwnedGeneration";

export function ImageGenerator({
  initialPrompt = "",
  initialAspect = "16:9",
  onUseImage,
  useImageLabel = "Use image",
  onCloseModal,
  setShouldPreventDismiss,
}: {
  initialPrompt?: string;
  initialAspect?: ImageAspect;
  /** When set, a finished image shows this action. The caller decides where the image goes. */
  onUseImage?: (asset: ExploreAsset) => void;
  useImageLabel?: string;
  onCloseModal?: () => void;
  setShouldPreventDismiss?: (fn: (() => boolean) | undefined) => void;
}) {
  const runtime = useExploreRuntime();
  const providers = useImageProviderChoice();
  const {
    createError,
    generation,
    generationId,
    hasStarted,
    inFlight,
    output,
    start,
  } = useOwnedGeneration();
  const cancel = useCancelGeneration();
  const confirmDelete = useConfirmDeleteAction();
  const isCancelling =
    cancel.isPending ||
    generation?.status === "cancelling" ||
    (generation !== undefined &&
      cancel.isSuccess &&
      cancel.variables === generation.id);

  useEffect(() => {
    if (!setShouldPreventDismiss || !onCloseModal) return;
    setShouldPreventDismiss(() => {
      if (!inFlight || isCancelling) return false;
      confirmDelete(GENERATION_CANCEL_CONFIRMATION, onCloseModal);
      return true;
    });
    return () => setShouldPreventDismiss(undefined);
  }, [confirmDelete, inFlight, isCancelling, onCloseModal, setShouldPreventDismiss]);

  const [prompt, setPrompt] = useState(initialPrompt);
  const [history, setHistory] = useState<{
    entries: PromptHistoryEntry[];
    index: number;
  }>({
    entries: [],
    index: -1,
  });
  const [aspect, setAspect] = useState<ImageAspect>(initialAspect);
  const [resolutionChoice, setResolutionChoice] = useState<ImageResolution>(
    DEFAULT_IMAGE_RESOLUTION,
  );
  // A shape too wide for the choice shows a smaller resolution. The choice returns on a shape that fits.
  const resolution = fittingImageResolution(aspect, resolutionChoice);
  const enhance = useImageEnhance({
    prompt,
    promptHistory: history.entries,
    historyIndex: history.index,
    onPromptHistoryChange: (entries, index) => {
      setHistory({ entries, index });
      setPrompt(entries[index]?.text ?? prompt);
    },
    disabled: inFlight,
  });

  const { provider, blockedReason } = providers;

  // As in FeatureFormScreen: Generate waits for a manual Enhance, which replaces the prompt.
  const canGenerate =
    prompt.trim().length > 0 &&
    provider !== null &&
    blockedReason === null &&
    !inFlight &&
    !enhance.isEnhancing;
  const previewUrl = output ? runtime.mediaUrlForAsset(output) : null;
  // The preview column opens on the first Generate and stays, so the card does not change size.
  const showPreview = hasStarted;
  const failed = createError !== null || generation?.status === "failed";
  const showCancel = generation !== undefined && isInFlightGeneration(generation);

  const generate = () => {
    if (!canGenerate || provider === null) return;
    start(
      buildTextToImageRequest({
        prompt,
        provenance: resolvePromptProvenance(prompt, history.entries, history.index),
        size: imageSizeForAspect(aspect, resolution),
        provider,
      }),
    );
  };

  return (
    <div className={clsx(styles.container, showPreview && styles.withPreview)}>
      <div className={styles.controls}>
        <div className={styles.field}>
          <Text as="span" variant="body" size="md" className={styles.label}>
            Prompt
          </Text>
          <textarea
            className={styles.textarea}
            value={prompt}
            placeholder="Describe the image"
            aria-label="Image prompt"
            onChange={(event) => setPrompt(event.currentTarget.value)}
          />
          {enhance.showManualEnhance ? (
            <div className={styles.enhanceRow}>
              <EnhanceSplitButton
                label={enhanceActionLabel(enhance.enhanceProvider, enhance.isEnhancing)}
                onEnhance={() => void enhance.runEnhance()}
                disabled={!enhance.canEnhance}
                isLoading={enhance.isEnhancing}
                showProviderMenu={enhance.canToggleEnhanceProvider}
                provider={enhance.enhanceProvider}
                onProviderChange={
                  enhance.canToggleEnhanceProvider
                    ? enhance.setEnhanceProvider
                    : undefined
                }
              />
              {enhance.enhanceError ? (
                <span className={styles.error} role="alert">
                  {enhance.enhanceError}
                </span>
              ) : null}
            </div>
          ) : null}
        </div>

        <div className={styles.settings}>
          <ImageSizeFields
            aspect={aspect}
            resolution={resolution}
            onAspectChange={setAspect}
            onResolutionChange={setResolutionChoice}
          />
          {providers.statusFailed ? (
            <div className={styles.field}>
              <span className={styles.error} role="alert">
                Could not check the image providers.
              </span>
              <Button
                appearance="neutral"
                hierarchy="secondary"
                size="sm"
                label="Try again"
                onClick={providers.retryStatus}
              />
            </div>
          ) : null}
          {provider !== null ? (
            <div className={clsx(styles.sizeField, styles.providerField)}>
              <ImageProviderPicker
                options={providers.options}
                provider={provider}
                blockedReason={blockedReason}
                onPick={providers.pick}
                labelClassName={styles.label}
                errorClassName={styles.error}
                segmentClassName={styles.providerSegment}
                fullWidth
              />
            </div>
          ) : null}
        </div>

        {failed ? (
          <span className={styles.error} role="alert">
            {createError !== null
              ? imageCreateErrorMessage(createError, runtime.host)
              : generation
                ? imageFailureMessage(generation)
                : ""}
          </span>
        ) : null}
        {generation?.status === "cancelled" ? (
          <Text as="p" variant="body" size="md" className={styles.status}>
            Cancelled
          </Text>
        ) : null}
      </div>

      {showPreview ? (
        <div className={styles.field}>
          <Text as="span" variant="body" size="md" className={styles.label}>
            Image
          </Text>
          <div
            className={styles.previewFrame}
            style={{ "--image-ratio": aspectRatioNumber(aspect) } as CSSProperties}
          >
            {previewUrl ? (
              <img className={styles.preview} src={previewUrl} alt="Generated" />
            ) : null}
            {inFlight ? (
              <div className={styles.previewStatus}>
                <ActivityCircular size={28} />
                <Text as="span" variant="body" size="md">
                  {generation?.status === "queued" ? "Waiting in the queue" : "Generating"}
                </Text>
              </div>
            ) : previewUrl ? null : (
              <Text as="span" variant="body" size="md" className={styles.previewEmpty}>
                No image was made
              </Text>
            )}
          </div>
        </div>
      ) : null}

      <div className={styles.actions}>
        {showCancel ? (
          <Button
            appearance="neutral"
            hierarchy="secondary"
            size="md"
            label="Cancel"
            disabled={isCancelling}
            onClick={() =>
              confirmDelete(GENERATION_CANCEL_CONFIRMATION, () =>
                cancel.mutate(generation.id),
              )
            }
          />
        ) : null}
        {output && onUseImage && !inFlight ? (
          <Button
            appearance="neutral"
            hierarchy="secondary"
            size="md"
            label={useImageLabel}
            onClick={() => onUseImage(output)}
          />
        ) : null}
        <Button
          appearance="brand"
          hierarchy="primary"
          size="md"
          label={generationId === null ? "Generate" : "Regenerate"}
          isLoading={inFlight}
          disabled={!canGenerate}
          onClick={generate}
        />
      </div>
    </div>
  );
}
