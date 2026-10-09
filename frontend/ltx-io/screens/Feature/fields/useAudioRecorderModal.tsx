import { useCallback } from "react";

import { AudioRecorder } from "@/ltx-io/components/AudioRecorder/AudioRecorder";
import { useModalContext } from "@/ltx-io/components/shared/Modal/ModalContext";
import type { MicrophoneAccess } from "@/ltx-io/media/microphone/types";

export const RECORD_AUDIO_MODAL_NAME = "record-audio";

export function useAudioRecorderModal(): (input: {
  maxDurationSeconds: number;
  onUseAudio: (file: File) => void;
  access: MicrophoneAccess;
}) => void {
  const { showModal, hideModal } = useModalContext();

  return useCallback(
    ({ maxDurationSeconds, onUseAudio, access }) => {
      showModal({
        content: (
          <AudioRecorder
            access={access}
            maxDurationSeconds={maxDurationSeconds}
            onUseAudio={(file) => {
              hideModal("apply");
              onUseAudio(file);
            }}
            onCancel={() => hideModal("cancel")}
          />
        ),
        modalName: RECORD_AUDIO_MODAL_NAME,
        variant: "secondary",
        noPadding: true,
        disablePointerClickOutside: true,
      });
    },
    [hideModal, showModal],
  );
}
