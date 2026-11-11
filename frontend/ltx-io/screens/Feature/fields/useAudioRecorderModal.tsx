import { useCallback } from "react";

import { AudioRecorder } from "@/ltx-io/components/AudioRecorder/AudioRecorder";
import { useCardModal } from "@/ltx-io/components/shared/Modal/useCardModal";
import type { MicrophoneAccess } from "@/ltx-io/media/microphone/types";

export const RECORD_AUDIO_MODAL_NAME = "record-audio";

export function useAudioRecorderModal(): (input: {
  maxDurationSeconds: number;
  onUseAudio: (file: File) => void;
  access: MicrophoneAccess;
}) => void {
  const showCardModal = useCardModal(RECORD_AUDIO_MODAL_NAME);

  return useCallback(
    ({ maxDurationSeconds, onUseAudio, access }) => {
      showCardModal(({ apply, cancel }) => (
        <AudioRecorder
          access={access}
          maxDurationSeconds={maxDurationSeconds}
          onUseAudio={(file) => {
            apply();
            onUseAudio(file);
          }}
          onCancel={cancel}
        />
      ));
    },
    [showCardModal],
  );
}
