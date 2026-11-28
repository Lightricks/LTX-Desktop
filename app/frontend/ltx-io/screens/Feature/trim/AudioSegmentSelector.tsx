import { Text } from "@ds/Text/Text";
import { useRef } from "react";

import type { ExploreAsset } from "@/lib/explore-contract";
import { AUDIO_WAVEFORM_STYLE } from "../../../components/AudioPlayer/audioWaveformStyle";
import { WaveForm } from "../../../components/WaveForm/WaveForm";
import { getWaveformPixelRatio } from "../../../components/WaveForm/waveformUtils";
import { TrimControls } from "./TrimControls.tsx";
import { TrimRange } from "./TrimRange.tsx";
import {
  CLIP_HEIGHT_AUDIO_COMPACT,
  WAVEFORM_VERTICAL_INSET,
} from "./trimConstants.ts";
import type { TrimRange as TrimRangeValue } from "./trimGeometry.ts";
import { useAudioSourceUrl } from "./useAudioSourceUrl.ts";
import { useTrimPlayback } from "./useTrimPlayback.ts";
import styles from "./AudioSegmentSelector.module.scss";

/**
 * Ported from LTX.io `AudioSegmentSelector`: an `<audio>` element that only
 * plays the selected In/Out window, a waveform trim track, and the transport
 * row underneath. The selection is controlled by the caller.
 */
export function AudioSegmentSelector({
  asset,
  durationSeconds,
  maxDurationSeconds,
  minDurationSeconds,
  range,
  onRangeChange,
}: {
  asset: ExploreAsset;
  durationSeconds: number;
  maxDurationSeconds: number;
  minDurationSeconds?: number;
  range: TrimRangeValue;
  onRangeChange: (range: TrimRangeValue) => void;
}) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const source = useAudioSourceUrl(asset);
  const playback = useTrimPlayback(audioRef, range);

  const waveformPixelRatio = getWaveformPixelRatio();

  return (
    <div className={styles.container}>
      <audio
        ref={audioRef}
        src={source.status === "ready" ? source.url : undefined}
        className={styles.audioElement}
        preload="metadata"
        {...playback.mediaHandlers}
      />

      <TrimRange
        durationSeconds={durationSeconds}
        maxDurationSeconds={maxDurationSeconds}
        minDurationSeconds={minDurationSeconds}
        range={range}
        onRangeChange={onRangeChange}
        mediaRef={audioRef}
        isPlaying={playback.isPlaying}
        onSeek={playback.setCurrentSec}
      >
        {({ clipWidth, clipHeight }) =>
          source.status === "ready" ? (
            <WaveForm
              src={source.url}
              width={clipWidth}
              height={clipHeight}
              startTime={0}
              endTime={durationSeconds}
              pixelRatio={waveformPixelRatio}
              styleOverrides={{
                ...AUDIO_WAVEFORM_STYLE,
                maxBarHeight: Math.max(
                  AUDIO_WAVEFORM_STYLE.minBarHeight,
                  clipHeight - WAVEFORM_VERTICAL_INSET * 2,
                ),
              }}
            />
          ) : (
            <div
              className={styles.trackPlaceholder}
              style={{ height: CLIP_HEIGHT_AUDIO_COMPACT }}
            />
          )
        }
      </TrimRange>

      <TrimControls
        isPlaying={playback.isPlaying}
        currentSec={playback.currentSec}
        durationSec={durationSeconds}
        isDisabled={source.status !== "ready"}
        toggleIsPlaying={playback.toggleIsPlaying}
        inSec={range.startSec}
        outSec={range.endSec}
      />

      {source.status === "error" ? (
        <Text as="span" variant="body" size="sm" className={styles.sourceError}>
          Could not load this audio for preview. Trimming still works.
        </Text>
      ) : null}
    </div>
  );
}
