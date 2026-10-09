import { type ReactNode, useCallback } from "react";

import {
  type CloseReason,
  useModalContext,
} from "../../../components/shared/Modal/ModalContext";
import { useExploreRuntime } from "../../../runtime/ExploreRuntime";
import { audioTrimErrorMessage } from "../fields/audioAssetInput.ts";
import type { MediaTrimRequest } from "../fields/prepareDurationLimitedMediaImport.ts";
import { videoTrimErrorMessage } from "../fields/videoAssetInput.ts";

import { AudioSegmentSelector } from "./AudioSegmentSelector.tsx";
import { TrimMediaModal } from "./TrimMediaModal.tsx";
import { VideoSegmentSelector } from "./VideoSegmentSelector.tsx";
import {
  MIN_VIDEO_TRIM_DURATION_SECONDS,
  TRIM_AUDIO_MODAL_NAME,
  TRIM_VIDEO_MODAL_NAME,
} from "./trimConstants.ts";
import { createTrimModalSave } from "./trimModalSave.ts";

/** Opens a trim modal for an import request; Done runs the request's own trim. */
export type OpenTrimModal = (
  request: MediaTrimRequest,
  options?: { onCancel?: () => void },
) => void;

type ModalBody = (props: {
  request: MediaTrimRequest;
  save: (startSec: number, endSec: number) => Promise<void>;
  dismiss: (reason: CloseReason) => void;
}) => ReactNode;

function useOpenTrimModal(modalName: string, body: ModalBody): OpenTrimModal {
  const { showModal, hideModal } = useModalContext();

  return useCallback(
    (request, openOptions) => {
      const { save, isSaving } = createTrimModalSave(request, hideModal);
      showModal({
        content: body({ request, save, dismiss: hideModal }),
        modalName,
        variant: "primary",
        shouldPreventDismiss: isSaving,
        onClose: (reason) => {
          if (reason !== "apply") openOptions?.onCancel?.();
        },
      });
    },
    [showModal, hideModal, modalName, body],
  );
}

export function useTrimVideoModal(): OpenTrimModal {
  const { mediaUrlForAsset } = useExploreRuntime();
  const body = useCallback<ModalBody>(
    ({ request, save, dismiss }) => {
      const src = mediaUrlForAsset(request.asset);
      const minDurationSeconds = Math.min(
        MIN_VIDEO_TRIM_DURATION_SECONDS,
        request.capSeconds,
      );
      return (
        <TrimMediaModal
          mediaKind="video"
          durationSeconds={request.durationSeconds}
          maxDurationSeconds={request.capSeconds}
          minDurationSeconds={minDurationSeconds}
          subtitle={request.subtitle}
          // Retake/extend only need the source under the cap, so open on the
          // widest window rather than a third of the clip.
          initialRange={{
            startSec: 0,
            endSec: Math.min(request.capSeconds, request.durationSeconds),
          }}
          saveErrorMessage={videoTrimErrorMessage}
          onSave={save}
          dismiss={dismiss}
          renderSelector={({ range, onRangeChange }) => (
            <VideoSegmentSelector
              src={src}
              durationSeconds={request.durationSeconds}
              maxDurationSeconds={request.capSeconds}
              minDurationSeconds={minDurationSeconds}
              range={range}
              onRangeChange={onRangeChange}
            />
          )}
        />
      );
    },
    [mediaUrlForAsset],
  );
  return useOpenTrimModal(TRIM_VIDEO_MODAL_NAME, body);
}

export function useTrimAudioModal(
  options: { minDurationSeconds?: number } = {},
): OpenTrimModal {
  const { minDurationSeconds } = options;
  const body = useCallback<ModalBody>(
    ({ request, save, dismiss }) => (
      <TrimMediaModal
        mediaKind="audio"
        durationSeconds={request.durationSeconds}
        maxDurationSeconds={request.capSeconds}
        minDurationSeconds={minDurationSeconds}
        saveErrorMessage={audioTrimErrorMessage}
        onSave={save}
        dismiss={dismiss}
        renderSelector={({ range, onRangeChange }) => (
          <AudioSegmentSelector
            asset={request.asset}
            durationSeconds={request.durationSeconds}
            maxDurationSeconds={request.capSeconds}
            minDurationSeconds={minDurationSeconds}
            range={range}
            onRangeChange={onRangeChange}
          />
        )}
      />
    ),
    [minDurationSeconds],
  );
  return useOpenTrimModal(TRIM_AUDIO_MODAL_NAME, body);
}
