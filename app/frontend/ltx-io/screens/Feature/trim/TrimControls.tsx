import { Button } from "@ds/Button/Button";
import { Text } from "@ds/Text/Text";
import PauseIcon from "@ds/assets/Icons/Pause.svg?react";
import PlayIcon from "@ds/assets/Icons/Play.svg?react";

import { formatTrimTimecode } from "./trimGeometry.ts";
import styles from "./TrimControls.module.scss";

/** Quiet label + emphasized time, ported from LTX.io `AudioControls`. */
function TimeMeta({ label, seconds }: { label: string; seconds: number }) {
  return (
    <span className={styles.timeMeta}>
      <Text as="span" variant="body" size="sm" className={styles.timeMetaLabel}>
        {label}
      </Text>
      <Text as="span" variant="body" size="sm" className={styles.timeMetaValue}>
        {formatTrimTimecode(seconds)}
      </Text>
    </span>
  );
}

export function TrimControls({
  isPlaying,
  currentSec,
  durationSec,
  isDisabled,
  toggleIsPlaying,
  inSec,
  outSec,
}: {
  isPlaying: boolean;
  currentSec: number;
  durationSec: number;
  isDisabled: boolean;
  toggleIsPlaying: () => void;
  inSec: number;
  outSec: number;
}) {
  return (
    <div className={styles.controlsRow}>
      <div className={styles.playbackGroup}>
        <Button
          appearance="neutral"
          hierarchy="plain"
          size="sm"
          isIconOnly
          disabled={isDisabled}
          leftIcon={isPlaying ? <PauseIcon /> : <PlayIcon />}
          aria-label={isPlaying ? "Pause" : "Play"}
          onClick={toggleIsPlaying}
        />
        <Text as="span" variant="body" size="sm" className={styles.timeDisplay}>
          {`${formatTrimTimecode(currentSec)} / ${formatTrimTimecode(durationSec)}`}
        </Text>
      </div>

      <div className={styles.rangeLabels} aria-live="polite">
        <TimeMeta label="In" seconds={inSec} />
        <TimeMeta label="Out" seconds={outSec} />
      </div>
    </div>
  );
}
