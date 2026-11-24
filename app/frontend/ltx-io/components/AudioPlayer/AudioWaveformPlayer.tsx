import { ActivityCircular } from "@ds/ActivityCircular/ActivityCircular";
import { Text } from "@ds/Text/Text";
import PauseIcon from "@ds/assets/Icons/Pause.svg?react";
import PlayIcon from "@ds/assets/Icons/Play.svg?react";
import { clsx } from "clsx";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type MouseEvent,
  type PointerEvent,
} from "react";

import { AUDIO_WAVEFORM_STYLE } from "@/ltx-io/components/AudioPlayer/audioWaveformStyle";
import { ProgressWaveForm } from "@/ltx-io/components/WaveForm/ProgressWaveForm";
import type { WaveformRenderState } from "@/ltx-io/components/WaveForm/useWaveform";
import { getWaveformPixelRatio } from "@/ltx-io/components/WaveForm/waveformUtils";

import styles from "./AudioWaveformPlayer.module.scss";
import { formatDurationToMMSS, useAudioPlayer } from "./useAudioPlayer";

const WAVEFORM_HEIGHT = 36;

export function AudioWaveformPlayer({
  src,
  disabled,
}: {
  src: string;
  disabled?: boolean;
}) {
  const waveformRef = useRef<HTMLDivElement>(null);
  const scrubPointerIdRef = useRef<number | null>(null);
  const pendingScrubClientXRef = useRef<number | null>(null);
  const scrubRafRef = useRef<number | null>(null);
  const [waveformWidth, setWaveformWidth] = useState(0);
  const [waveformState, setWaveformState] =
    useState<WaveformRenderState>("idle");
  const [isScrubbing, setIsScrubbing] = useState(false);
  const waveformPixelRatio = getWaveformPixelRatio();

  const {
    isPlaying,
    isLoading: isLoadingAudio,
    play,
    pause,
    seek,
    duration,
    currentTime,
  } = useAudioPlayer(src, {
    shouldLoadSrc: !!src,
  });

  useEffect(() => {
    setWaveformState("idle");
  }, [src]);

  useEffect(() => {
    if (!waveformRef.current || typeof ResizeObserver === "undefined") {
      return;
    }

    const observer = new ResizeObserver(([entry]) => {
      setWaveformWidth(Math.floor(entry.contentRect.width));
    });
    observer.observe(waveformRef.current);

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    return () => {
      if (scrubRafRef.current !== null) {
        cancelAnimationFrame(scrubRafRef.current);
      }
    };
  }, []);

  const handleTogglePlayback = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      event.stopPropagation();

      if (disabled) return;

      if (isPlaying) {
        pause();
      } else {
        void play();
      }
    },
    [disabled, isPlaying, pause, play],
  );

  const seekFromClientX = useCallback(
    (clientX: number) => {
      if (!waveformRef.current || !duration || disabled) return;

      const rect = waveformRef.current.getBoundingClientRect();
      const ratio = Math.min(
        Math.max((clientX - rect.left) / rect.width, 0),
        1,
      );
      seek(ratio * duration);
    },
    [disabled, duration, seek],
  );

  const flushPendingScrub = useCallback(() => {
    scrubRafRef.current = null;
    const clientX = pendingScrubClientXRef.current;
    pendingScrubClientXRef.current = null;
    if (clientX === null || scrubPointerIdRef.current === null) return;
    seekFromClientX(clientX);
  }, [seekFromClientX]);

  const scheduleScrubSeek = useCallback(
    (clientX: number) => {
      pendingScrubClientXRef.current = clientX;
      if (scrubRafRef.current !== null) return;
      scrubRafRef.current = requestAnimationFrame(flushPendingScrub);
    },
    [flushPendingScrub],
  );

  const handlePointerDown = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      event.preventDefault();
      event.stopPropagation();

      if (!duration || disabled || scrubPointerIdRef.current !== null) return;

      scrubPointerIdRef.current = event.pointerId;
      setIsScrubbing(true);
      event.currentTarget.setPointerCapture(event.pointerId);
      seekFromClientX(event.clientX);
    },
    [disabled, duration, seekFromClientX],
  );

  const handlePointerMove = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (scrubPointerIdRef.current !== event.pointerId) return;

      event.preventDefault();
      scheduleScrubSeek(event.clientX);
    },
    [scheduleScrubSeek],
  );

  const endScrubbing = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (scrubPointerIdRef.current !== event.pointerId) return;

      if (scrubRafRef.current !== null) {
        cancelAnimationFrame(scrubRafRef.current);
        scrubRafRef.current = null;
      }
      const pendingX = pendingScrubClientXRef.current;
      pendingScrubClientXRef.current = null;
      if (pendingX !== null) {
        seekFromClientX(pendingX);
      }

      scrubPointerIdRef.current = null;
      setIsScrubbing(false);

      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
    },
    [seekFromClientX],
  );

  const currentLabel = formatDurationToMMSS(
    Number.isFinite(currentTime) ? Math.max(currentTime, 0) : 0,
  );
  const durationLabel = formatDurationToMMSS(
    Number.isFinite(duration) ? Math.max(duration, 0) : 0,
  );

  const shouldShowLoader =
    !src || isLoadingAudio || duration === 0 || waveformWidth === 0;
  const progress = duration > 0 ? currentTime / duration : 0;
  const showWaveform = waveformState !== "error";

  return (
    <div className={clsx(styles.container, disabled && styles.disabled)}>
      <button
        type="button"
        className={styles.playButton}
        onClick={handleTogglePlayback}
        disabled={disabled || shouldShowLoader}
        aria-label={isPlaying ? "Pause audio" : "Play audio"}
      >
        {shouldShowLoader ? (
          <ActivityCircular size={16} />
        ) : isPlaying ? (
          <PauseIcon />
        ) : (
          <PlayIcon />
        )}
      </button>

      <div className={styles.content}>
        <div
          ref={waveformRef}
          className={clsx(
            styles.waveformContainer,
            isScrubbing && styles.scrubbing,
          )}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={endScrubbing}
          onPointerCancel={endScrubbing}
          onClick={(event) => event.stopPropagation()}
        >
          {shouldShowLoader || !showWaveform ? (
            <div className={styles.waveformPlaceholder} />
          ) : (
            <>
              <ProgressWaveForm
                src={src}
                width={waveformWidth}
                height={WAVEFORM_HEIGHT}
                startTime={0}
                endTime={duration}
                progress={progress}
                onStateChange={setWaveformState}
                pixelRatio={waveformPixelRatio}
                styleOverrides={{
                  ...AUDIO_WAVEFORM_STYLE,
                  maxBarHeight: WAVEFORM_HEIGHT - 4,
                }}
              />
              <div
                className={styles.playhead}
                style={{ left: `${progress * 100}%` }}
                aria-hidden
              />
            </>
          )}
        </div>

        <div className={styles.timeRow}>
          <Text variant="label" size="xs" as="span" className={styles.timeText}>
            {currentLabel}
          </Text>
          <Text variant="label" size="xs" as="span" className={styles.timeText}>
            {durationLabel}
          </Text>
        </div>
      </div>
    </div>
  );
}
