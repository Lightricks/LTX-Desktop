import { Text } from "@ds/Text/Text";
import type { RefObject } from "react";

import formStyles from "../FeatureFormView.module.scss";
import { TrimRange } from "../trim/TrimRange";
import { VideoFilmstrip } from "../trim/VideoFilmstrip.tsx";
import { CLIP_HEIGHT_AUDIO_COMPACT } from "../trim/trimConstants";
import styles from "./VideoAssetField.module.scss";
import type { VideoAssetRangeValue } from "./VideoAssetPreview";

export function VideoTrimSection({
  src,
  durationSeconds,
  range,
  videoRef,
  isPlaying,
}: {
  src: string;
  durationSeconds: number;
  range: VideoAssetRangeValue;
  videoRef: RefObject<HTMLVideoElement>;
  isPlaying: boolean;
}) {
  const startSec = range.startSec;
  const endSec = range.startSec + range.durationSec;

  return (
    <div className={styles.range}>
      <Text as="span" variant="body" size="md" className={formStyles.fieldLabel}>
        {range.label}
      </Text>
      <TrimRange
        durationSeconds={durationSeconds}
        maxDurationSeconds={Math.min(range.maxDurationSec, durationSeconds)}
        minDurationSeconds={range.minDurationSec}
        range={{ startSec, endSec }}
        onRangeChange={(next) =>
          range.onChange(next.startSec, next.endSec - next.startSec)
        }
        mediaRef={videoRef as RefObject<HTMLMediaElement>}
        isPlaying={isPlaying}
        playheadRange={{ startSec: 0, endSec: durationSeconds }}
        clipHeight={CLIP_HEIGHT_AUDIO_COMPACT}
        labelPlacement="bottom"
      >
        {({ clipWidth, clipHeight }) => (
          <VideoFilmstrip
            src={src}
            width={clipWidth}
            height={clipHeight}
            durationSeconds={durationSeconds}
          />
        )}
      </TrimRange>
    </div>
  );
}
