import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import { Tooltip } from "@ds/Tooltip/Tooltip";
import PauseIcon from "@ds/assets/Icons/Pause.svg?react";
import PlayIcon from "@ds/assets/Icons/Play.svg?react";
import RecordIcon from "@ds/assets/Icons/Record.svg?react";
import RemoveIcon from "@ds/assets/Icons/Remove.svg?react";
import StopIcon from "@ds/assets/Icons/Stop.svg?react";
import { clsx } from "clsx";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type RefObject,
} from "react";

import { formatDurationToMMSS } from "@/ltx-io/components/AudioPlayer/useAudioPlayer";
import { RoundCloseButton } from "@/ltx-io/components/shared/RoundCloseButton/RoundCloseButton";
import { micRemedyFor, micRemedyLabel } from "@/ltx-io/media/microphone/remedy";
import type {
  LiveMicrophoneAccess,
  MicBlocker,
} from "@/ltx-io/media/microphone/types";
import { recordedAudioToWav } from "@/ltx-io/media/microphone/encodePcmWav";

import styles from "./AudioRecorder.module.scss";
import { AudioVisualizer } from "./AudioVisualizer/AudioVisualizer";
import {
  useAudioPlayback,
  useAudioRecording,
  useCountdown,
} from "./hooks";

const COUNTDOWN_SECONDS = 3;

type AudioRecorderState =
  | "idle"
  | "requesting"
  | "blocked"
  | "countdown"
  | "recording"
  | "encoding"
  | "stopped";

const getVisualizerBarColor = (
  recordingDisplayRef: RefObject<HTMLDivElement | null>,
) => {
  if (!recordingDisplayRef.current) return undefined;
  return getComputedStyle(recordingDisplayRef.current).color;
};

export function AudioRecorder({
  onUseAudio,
  onCancel,
  maxDurationSeconds,
  access,
}: {
  onUseAudio: (audioFile: File) => void;
  onCancel?: () => void;
  maxDurationSeconds: number;
  access: LiveMicrophoneAccess;
}) {
  const [state, setState] = useState<AudioRecorderState>("idle");
  const [blocker, setBlocker] = useState<MicBlocker | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [previewBuffer, setPreviewBuffer] = useState<AudioBuffer | null>(null);
  const [recordedDurationSeconds, setRecordedDurationSeconds] = useState(0);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const recordingDisplayRef = useRef<HTMLDivElement>(null);
  const captureGenerationRef = useRef(0);

  const stopStream = useCallback((current: MediaStream | null) => {
    current?.getTracks().forEach((track) => track.stop());
  }, []);

  const playback = useAudioPlayback(previewBuffer, recordedDurationSeconds);
  const { reset: resetPlayback } = playback;

  const handleRecordingFailed = useCallback(() => {
    setStream((current) => {
      stopStream(current);
      return null;
    });
    setBlocker({
      cause: "unreadable-recording",
      remedy: micRemedyFor(
        "unreadable-recording",
        access.openMicrophoneSettings != null,
      ),
    });
    setState("blocked");
  }, [access.openMicrophoneSettings, stopStream]);

  const handleRecordingComplete = useCallback(
    (file: File) => {
      const generation = captureGenerationRef.current;
      setStream((current) => {
        stopStream(current);
        return null;
      });
      setState("encoding");
      void recordedAudioToWav(file)
        .then(({ wav, buffer }) => {
          if (generation !== captureGenerationRef.current) {
            return;
          }
          setAudioFile(wav);
          setPreviewBuffer(buffer);
          setState("stopped");
        })
        .catch(() => {
          if (generation !== captureGenerationRef.current) {
            return;
          }
          handleRecordingFailed();
        });
    },
    [handleRecordingFailed, stopStream],
  );

  const recording = useAudioRecording({
    stream,
    maxDurationSeconds,
    onRecordingComplete: handleRecordingComplete,
    onRecordingFailed: handleRecordingFailed,
    onAutoStop: () => {
      setRecordedDurationSeconds(maxDurationSeconds);
    },
  });
  const { abortRecording, startRecording } = recording;

  const handleCountdownComplete = useCallback(() => {
    if (startRecording()) {
      setState("recording");
      return;
    }
    setStream((current) => {
      stopStream(current);
      return null;
    });
    setBlocker({
      cause: "capture-unsupported",
      remedy: micRemedyFor(
        "capture-unsupported",
        access.openMicrophoneSettings != null,
      ),
    });
    setState("blocked");
  }, [access.openMicrophoneSettings, startRecording, stopStream]);

  const countdown = useCountdown({
    initialSeconds: COUNTDOWN_SECONDS,
    onComplete: handleCountdownComplete,
  });
  const { abortCountdown, startCountdown } = countdown;

  const pendingCountdownRef = useRef(false);

  const resetRecordingState = useCallback(() => {
    captureGenerationRef.current += 1;
    pendingCountdownRef.current = false;
    abortCountdown();
    abortRecording();
    resetPlayback();
    setAudioFile(null);
    setPreviewBuffer(null);
    setRecordedDurationSeconds(0);
    setBlocker(null);
    setStream((current) => {
      stopStream(current);
      return null;
    });
  }, [abortCountdown, abortRecording, resetPlayback, stopStream]);

  useEffect(() => {
    return () => {
      captureGenerationRef.current += 1;
    };
  }, []);

  useEffect(() => {
    return () => {
      stopStream(stream);
    };
  }, [stopStream, stream]);

  const beginLiveCapture = useCallback(async () => {
    const generation = ++captureGenerationRef.current;
    setStream((current) => {
      stopStream(current);
      return null;
    });
    setState("requesting");
    setBlocker(null);
    try {
      const result = await access.requestStream();
      if (generation !== captureGenerationRef.current) {
        if (result.status === "granted") {
          stopStream(result.stream);
        }
        return;
      }
      if (result.status === "blocked") {
        setBlocker(result.blocker);
        setState("blocked");
        return;
      }
      pendingCountdownRef.current = true;
      setStream(result.stream);
      setState("countdown");
    } catch {
      if (generation !== captureGenerationRef.current) {
        return;
      }
      setStream((current) => {
        stopStream(current);
        return null;
      });
      setBlocker({
        cause: "unknown",
        remedy: micRemedyFor(
          "unknown",
          access.openMicrophoneSettings != null,
        ),
      });
      setState("blocked");
    }
  }, [access, stopStream]);

  useEffect(() => {
    if (state !== "countdown" || stream == null || !pendingCountdownRef.current) {
      return;
    }
    pendingCountdownRef.current = false;
    startCountdown();
  }, [startCountdown, state, stream]);

  const handleStartRecording = useCallback(() => {
    void beginLiveCapture();
  }, [beginLiveCapture]);

  const handleStopRecording = useCallback(() => {
    setRecordedDurationSeconds(recording.elapsedSeconds);
    recording.stopRecording();
  }, [recording]);

  const handleDiscard = useCallback(() => {
    resetRecordingState();
    setState("idle");
  }, [resetRecordingState]);

  const handleCancel = useCallback(() => {
    resetRecordingState();
    setState("idle");
    onCancel?.();
  }, [onCancel, resetRecordingState]);

  const handleUseAudio = useCallback(() => {
    if (audioFile) {
      onUseAudio(audioFile);
    }
  }, [audioFile, onUseAudio]);

  const handleOpenMicrophoneSettings = useCallback(async () => {
    if (access.openMicrophoneSettings == null) {
      return;
    }
    try {
      await access.openMicrophoneSettings();
    } catch {
      setBlocker({
        cause: "os-denied",
        remedy: {
          title: "Couldn't open Settings.",
          body: "Open microphone privacy settings yourself, then try again.",
          action: "open-os-settings",
        },
      });
      setState("blocked");
    }
  }, [access.openMicrophoneSettings]);

  const handleBlockedAction = useCallback(() => {
    if (blocker == null) {
      return;
    }
    if (blocker.remedy.action === "open-os-settings") {
      void handleOpenMicrophoneSettings();
      return;
    }
    if (blocker.remedy.action === "retry") {
      void beginLiveCapture();
    }
  }, [beginLiveCapture, blocker, handleOpenMicrophoneSettings]);

  const playbackReady =
    state === "stopped" && audioFile != null && playback.isReady;
  const isPreparing = state === "encoding";
  const stoppedDurationSeconds =
    playback.durationSeconds > 0
      ? playback.durationSeconds
      : recordedDurationSeconds;
  const displaySeconds =
    state === "stopped" || state === "encoding"
      ? playback.progress === 0
        ? stoppedDurationSeconds
        : playback.progress * stoppedDurationSeconds
      : recording.elapsedSeconds;

  const isVisualizerActive =
    stream != null &&
    state !== "idle" &&
    state !== "requesting" &&
    state !== "blocked";
  const barColor = getVisualizerBarColor(recordingDisplayRef) || undefined;
  const subtitle =
    state === "blocked" && blocker != null
      ? blocker.remedy.body
      : `Up to ${maxDurationSeconds} seconds`;

  return (
    <div className={styles.container}>
      <RoundCloseButton className={styles.closeButton} onClick={handleCancel} />
      <div className={styles.header}>
        <Text variant="heading" size="sm" as="h2" align="center">
          {state === "blocked" && blocker != null
            ? blocker.remedy.title
            : "Record Audio"}
        </Text>
        <Text
          variant="body"
          size="lg"
          as="p"
          align="center"
          className={styles.subtitle}
        >
          {subtitle}
        </Text>
      </div>
      <div ref={recordingDisplayRef} className={styles.recordingDisplay}>
        {playbackReady ? (
          <div className={styles.visualizerContainer}>
            <AudioVisualizer
              source="analyser"
              analyser={playback.analyser}
              isPlaying={playback.isPlaying}
              barColor={barColor}
            />
          </div>
        ) : (
          <div className={styles.visualizerContainer}>
            <AudioVisualizer
              source="stream"
              stream={stream}
              isActive={isVisualizerActive}
              barColor={barColor}
            />
          </div>
        )}
        <div className={styles.timerWrapper}>
          <Text
            variant="heading"
            size="xxl"
            as="span"
            className={clsx(styles.timer, styles.timerMonospace)}
          >
            {formatDurationToMMSS(displaySeconds)}
          </Text>
        </div>
      </div>
      {state === "blocked" && blocker != null ? (
        <div className={styles.controlsRow}>
          <Button
            appearance="neutral"
            hierarchy="secondary"
            size="md"
            label="Cancel"
            onClick={handleCancel}
          />
          {blocker.remedy.action === "open-os-settings" ? (
            <>
              <Button
                appearance="neutral"
                hierarchy="secondary"
                size="md"
                label="Open Settings"
                onClick={() => {
                  void handleOpenMicrophoneSettings();
                }}
              />
              <Button
                appearance="danger"
                hierarchy="primary"
                size="md"
                label="Try again"
                leftIcon={<RecordIcon />}
                onClick={() => {
                  void beginLiveCapture();
                }}
              />
            </>
          ) : blocker.remedy.action !== "none" ? (
            <Button
              appearance="danger"
              hierarchy="primary"
              size="md"
              label={micRemedyLabel(blocker.remedy.action)}
              leftIcon={<RecordIcon />}
              onClick={handleBlockedAction}
            />
          ) : null}
        </div>
      ) : null}
      {(state === "idle" || state === "requesting") ? (
        <div className={styles.controlsRow}>
          <Button
            appearance="danger"
            hierarchy="primary"
            size="md"
            label="Start Recording"
            leftIcon={<RecordIcon />}
            disabled={state === "requesting"}
            onClick={handleStartRecording}
          />
        </div>
      ) : null}
      {state === "countdown" ? (
        <div className={styles.controlsRow}>
          <Button
            appearance="danger"
            hierarchy="primary"
            size="md"
            label={`Starting in ${countdown.secondsRemaining}`}
            leftIcon={<RecordIcon />}
            disabled
          />
        </div>
      ) : null}
      {state === "recording" ? (
        <div className={styles.controlsRow}>
          <div className={styles.recordingControls}>
            <Tooltip content="Start over">
              <Button
                appearance="neutral"
                hierarchy="secondary"
                size="md"
                isIconOnly
                leftIcon={<RemoveIcon />}
                aria-label="Start over"
                onClick={handleDiscard}
              />
            </Tooltip>
            <Tooltip content="Stop recording">
              <Button
                appearance="neutral"
                hierarchy="secondary"
                size="md"
                isIconOnly
                leftIcon={<StopIcon />}
                aria-label="Stop recording"
                onClick={handleStopRecording}
              />
            </Tooltip>
          </div>
        </div>
      ) : null}
      {isPreparing ? (
        <div className={styles.controlsRow}>
          <Button
            appearance="danger"
            hierarchy="primary"
            size="md"
            label="Preparing…"
            disabled
          />
        </div>
      ) : null}
      {playbackReady ? (
        <div className={clsx(styles.controlsRow, styles.stoppedControlsRow)}>
          <div className={styles.playbackControls}>
            <div className={styles.seekContainer}>
              <Button
                appearance="neutral"
                hierarchy="plain"
                size="md"
                isIconOnly
                leftIcon={playback.isPlaying ? <PauseIcon /> : <PlayIcon />}
                aria-label={playback.isPlaying ? "Pause" : "Play"}
                onClick={playback.togglePlayback}
              />
              <div
                className={styles.seekBar}
                onClick={(event) => {
                  const rect = event.currentTarget.getBoundingClientRect();
                  playback.seek(
                    Math.max(
                      0,
                      Math.min(1, (event.clientX - rect.left) / rect.width),
                    ),
                  );
                }}
              >
                <div
                  className={styles.seekProgress}
                  style={{ width: `${playback.progress * 100}%` }}
                />
                <div className={styles.seekTrack} />
              </div>
            </div>
          </div>
          <div className={styles.stoppedControls}>
            <Tooltip content="Start over">
              <Button
                appearance="neutral"
                hierarchy="secondary"
                size="md"
                isIconOnly
                leftIcon={<RemoveIcon />}
                aria-label="Start over"
                onClick={handleDiscard}
              />
            </Tooltip>
            <Button
              appearance="brand"
              hierarchy="primary"
              size="md"
              label="Use audio"
              onClick={handleUseAudio}
            />
          </div>
        </div>
      ) : null}
    </div>
  );
}
