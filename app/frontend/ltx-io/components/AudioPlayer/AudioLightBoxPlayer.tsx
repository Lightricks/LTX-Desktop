import { useCallback, useEffect, useRef, useState, type MouseEvent } from "react";

import { ActivityCircular } from "@ds/ActivityCircular/ActivityCircular";

import { AUDIO_WAVEFORM_STYLE } from "@/ltx-io/components/AudioPlayer/audioWaveformStyle";
import {
  LIGHTBOX_MEDIA_CONTROLS,
  VideoControls,
} from "@/ltx-io/components/VideoView/VideoControls";
import { ProgressWaveForm } from "@/ltx-io/components/WaveForm/ProgressWaveForm";
import { getWaveformPixelRatio } from "@/ltx-io/components/WaveForm/waveformUtils";

import { audioLightboxWaveformHeight } from "./audioLightboxLayout";
import styles from "./AudioLightBoxPlayer.module.scss";
import { useAudioPlayer } from "./useAudioPlayer";

export function AudioLightBoxPlayer({
  src,
  autoPlay = false,
  muted = false,
  onMute,
  onError,
}: {
  src: string;
  autoPlay?: boolean;
  muted?: boolean;
  onMute?: () => void;
  onError?: () => void;
}) {
  const waveformRef = useRef<HTMLDivElement>(null);
  const [waveformWidth, setWaveformWidth] = useState(0);
  const [waveformAreaHeight, setWaveformAreaHeight] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const waveformPixelRatio = getWaveformPixelRatio();

  const { isPlaying, isLoading, play, pause, seek, duration, currentTime } =
    useAudioPlayer(src, {
      shouldLoadSrc: !!src,
      muted,
      autoPlay,
      onError,
    });

  useEffect(() => {
    const element = waveformRef.current;
    if (!element || typeof ResizeObserver === "undefined") return;

    const updateSize = (width: number, height: number) => {
      setWaveformWidth(Math.floor(width));
      setWaveformAreaHeight(Math.floor(height));
    };
    updateSize(element.clientWidth, element.clientHeight);

    const observer = new ResizeObserver(([entry]) => {
      updateSize(entry.contentRect.width, entry.contentRect.height);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const togglePlayback = useCallback(() => {
    if (isPlaying) {
      pause();
      return;
    }
    void play();
  }, [isPlaying, pause, play]);

  const handleToggle = useCallback(
    (event: MouseEvent) => {
      event.stopPropagation();
      togglePlayback();
    },
    [togglePlayback],
  );

  const handleSeek = useCallback(
    (newTime: number) => {
      seek(newTime);
      if (muted) onMute?.();
    },
    [muted, onMute, seek],
  );

  const progress = duration > 0 ? currentTime / duration : 0;
  const isAudioReady = !!src && !isLoading && duration > 0;
  const canShowWaveform = isAudioReady && waveformWidth > 0;
  const waveformHeight = audioLightboxWaveformHeight(waveformAreaHeight);

  const handleWaveformClick = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      event.stopPropagation();
      if (!isAudioReady) return;
      togglePlayback();
    },
    [isAudioReady, togglePlayback],
  );

  return (
    <div className={styles.outerWrapper} data-audio-player>
      <div
        ref={waveformRef}
        className={styles.waveformArea}
        onClick={handleWaveformClick}
      >
        {canShowWaveform ? (
          <ProgressWaveForm
            src={src}
            width={waveformWidth}
            height={waveformHeight}
            startTime={0}
            endTime={duration}
            pixelRatio={waveformPixelRatio}
            progress={progress}
            styleOverrides={{
              ...AUDIO_WAVEFORM_STYLE,
              maxBarHeight: waveformHeight - 10,
            }}
          />
        ) : (
          <div className={styles.placeholder}>
            <ActivityCircular />
          </div>
        )}
      </div>

      <VideoControls
        showElements={LIGHTBOX_MEDIA_CONTROLS}
        currentTime={currentTime}
        duration={duration}
        onSeekCallback={handleSeek}
        onDraggingStarted={() => setIsDragging(true)}
        onDraggingEnded={() => setIsDragging(false)}
        onMute={onMute}
        isDragging={isDragging}
        isPlaying={isPlaying}
        isMuted={muted}
        togglePlaying={handleToggle}
        onFullScreenClick={() => undefined}
        isFullScreen={false}
      />
    </div>
  );
}
