import { Text } from "@ds/Text/Text";
import { useRef, useState } from "react";

import { TrimControls } from "./TrimControls.tsx";
import { TrimRange } from "./TrimRange.tsx";
import { VideoFilmstrip } from "./VideoFilmstrip.tsx";
import type { TrimRange as TrimRangeValue } from "./trimGeometry.ts";
import { useTrimPlayback } from "./useTrimPlayback.ts";
import styles from "./VideoSegmentSelector.module.scss";

/**
 * Video counterpart of `AudioSegmentSelector`, ported from LTX.io
 * `VideoSegmentSelector`: a preview whose playhead roams the whole clip
 * (audio stays inside In/Out), a filmstrip trim track, and the transport row.
 */
export function VideoSegmentSelector({
  src,
  durationSeconds,
  maxDurationSeconds,
  minDurationSeconds,
  range,
  onRangeChange,
}: {
  src: string | null;
  durationSeconds: number;
  maxDurationSeconds: number;
  minDurationSeconds?: number;
  range: TrimRangeValue;
  onRangeChange: (range: TrimRangeValue) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const fullClip = { startSec: 0, endSec: durationSeconds };
  const playback = useTrimPlayback(videoRef, fullClip);
  const [loadFailed, setLoadFailed] = useState(false);
  const canPlay = src != null && !loadFailed;

  return (
    <div className={styles.container}>
      <div className={styles.preview}>
        {src != null ? (
          <video
            ref={videoRef}
            src={src}
            className={styles.video}
            preload="metadata"
            playsInline
            onClick={playback.toggleIsPlaying}
            onError={() => setLoadFailed(true)}
            {...playback.mediaHandlers}
          />
        ) : null}
      </div>

      <TrimRange
        durationSeconds={durationSeconds}
        maxDurationSeconds={maxDurationSeconds}
        minDurationSeconds={minDurationSeconds}
        range={range}
        onRangeChange={onRangeChange}
        mediaRef={videoRef}
        isPlaying={playback.isPlaying}
        onSeek={playback.setCurrentSec}
        playheadRange={fullClip}
        labelPlacement="bottom"
      >
        {({ clipWidth, clipHeight }) =>
          src != null ? (
            <VideoFilmstrip
              src={src}
              width={clipWidth}
              height={clipHeight}
              durationSeconds={durationSeconds}
            />
          ) : (
            <div className={styles.trackPlaceholder} style={{ height: clipHeight }} />
          )
        }
      </TrimRange>

      <TrimControls
        isPlaying={playback.isPlaying}
        currentSec={playback.currentSec}
        durationSec={durationSeconds}
        isDisabled={!canPlay}
        toggleIsPlaying={playback.toggleIsPlaying}
        inSec={range.startSec}
        outSec={range.endSec}
      />

      {!canPlay ? (
        <Text as="span" variant="body" size="sm" className={styles.sourceError}>
          Could not load this video for preview. Trimming still works.
        </Text>
      ) : null}
    </div>
  );
}
