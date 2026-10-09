import { useEffect, useRef, useState } from "react";

import { Button } from "@ds/Button/Button";
import { Tooltip } from "@ds/Tooltip/Tooltip";
import MuteIcon from "@ds/assets/Icons/Audio/Mute.svg?react";
import UnmuteIcon from "@ds/assets/Icons/Audio/On.svg?react";

import { AUDIO_WAVEFORM_STYLE } from "@/ltx-io/components/AudioPlayer/audioWaveformStyle";
import { ProgressWaveForm } from "@/ltx-io/components/WaveForm/ProgressWaveForm";
import {
  getWaveformPixelRatio,
  releaseDecodedAudio,
} from "@/ltx-io/components/WaveForm/waveformUtils";
import type { ExploreListedAsset } from "@/lib/explore-contract";
import { useExploreRuntime } from "@/ltx-io/runtime/ExploreRuntime";
import { useFeatureFormMuteStore } from "@/ltx-io/stores/featureFormMuteStore";

import {
  audioPlaybackProgress,
  shouldPlayAudioPreview,
} from "./audioTilePreview";
import styles from "./MediaTile.module.scss";

function useElementWidth(): [
  HTMLDivElement | null,
  (element: HTMLDivElement | null) => void,
] {
  const [element, setElement] = useState<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    if (!element) return;
    const observer = new ResizeObserver(() => setWidth(element.clientWidth));
    observer.observe(element);
    setWidth(element.clientWidth);
    return () => observer.disconnect();
  }, [element]);

  return [width > 0 ? element : null, setElement];
}

function readPositiveDuration(audio: HTMLAudioElement): number {
  const next = audio.duration;
  return Number.isFinite(next) && next > 0 ? next : 0;
}

export function AssetsAudioTile({
  asset,
  isHovered,
  shouldHoverPlay,
}: {
  asset: ExploreListedAsset;
  isHovered: boolean;
  shouldHoverPlay: boolean;
}) {
  const { waveformUrlForAsset } = useExploreRuntime();
  const isMuted = useFeatureFormMuteStore((state) => state.isMuted);
  const setMuted = useFeatureFormMuteStore((state) => state.setMuted);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [waveformUrl, setWaveformUrl] = useState<string | null>(null);
  const [waveformElement, setWaveformElement] = useElementWidth();
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const shouldPlay = shouldPlayAudioPreview(isHovered, shouldHoverPlay);
  const waveformPixelRatio = getWaveformPixelRatio();

  useEffect(() => {
    let cancelled = false;
    let createdUrl: string | null = null;

    void waveformUrlForAsset({
      id: asset.id,
      path: asset.path,
      bytes_url: asset.bytes_url,
    })
      .then((url) => {
        if (!url?.startsWith("blob:")) return;
        if (cancelled) {
          URL.revokeObjectURL(url);
          return;
        }
        createdUrl = url;
        setWaveformUrl(url);
      })
      .catch(() => undefined);

    return () => {
      cancelled = true;
      setWaveformUrl(null);
      setCurrentTime(0);
      setDuration(0);
      if (createdUrl) {
        releaseDecodedAudio(createdUrl);
        URL.revokeObjectURL(createdUrl);
      }
    };
  }, [asset.bytes_url, asset.id, asset.path, waveformUrlForAsset]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !waveformUrl || !shouldPlay) return;

    audio.currentTime = 0;
    setCurrentTime(0);

    let frame = 0;
    const tick = () => {
      setCurrentTime(audio.currentTime);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    void audio.play().catch(() => undefined);

    return () => {
      cancelAnimationFrame(frame);
      audio.pause();
      audio.currentTime = 0;
      setCurrentTime(0);
    };
  }, [shouldPlay, waveformUrl]);

  return (
    <div className={styles.audio} ref={setWaveformElement}>
      {waveformElement && waveformUrl && duration > 0 ? (
        <ProgressWaveForm
          src={waveformUrl}
          width={waveformElement.clientWidth}
          height={48}
          startTime={0}
          endTime={duration}
          progress={audioPlaybackProgress(currentTime, duration)}
          pixelRatio={waveformPixelRatio}
          styleOverrides={AUDIO_WAVEFORM_STYLE}
          className={styles.waveform}
        />
      ) : (
        <div className={styles.placeholder} aria-label="Audio preview">
          <UnmuteIcon />
        </div>
      )}
      {waveformUrl ? (
        <audio
          ref={audioRef}
          src={waveformUrl}
          muted={isMuted}
          loop
          preload="metadata"
          onLoadedMetadata={(event) =>
            setDuration(readPositiveDuration(event.currentTarget))
          }
          onDurationChange={(event) =>
            setDuration(readPositiveDuration(event.currentTarget))
          }
        />
      ) : null}
      <div
        className={styles.mute}
        onClick={(event) => event.stopPropagation()}
      >
        <Tooltip content={isMuted ? "Unmute" : "Mute"}>
          <Button
            appearance="overlay"
            hierarchy="secondary"
            size="md"
            isIconOnly
            leftIcon={isMuted ? <MuteIcon /> : <UnmuteIcon />}
            aria-label={isMuted ? "Unmute audio" : "Mute audio"}
            onClick={() => setMuted(!isMuted)}
          />
        </Tooltip>
      </div>
    </div>
  );
}
