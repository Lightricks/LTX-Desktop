import { useId } from "react";

import { PLAYHEAD_HANDLE_WIDTH_PX } from "./trimConstants.ts";
import styles from "./TrimSeekBar.module.scss";

/**
 * LTX.io `SeekBar.svg` (13px shield head on a 3px stem), with the stem
 * stretched to `height` so one shape serves every track height.
 */
function needlePath(height: number): string {
  const stemEnd = height - 1.5;
  return [
    "M11 0C12.105 0 13 .895 13 2V7.952C13 8.608 12.678 9.223 12.139 9.597L8 12.461",
    `V${stemEnd}A1.5 1.5 0 0 1 5 ${stemEnd}`,
    "V12.461L.861 9.597C.322 9.223 0 8.608 0 7.952V2C0 .895 .895 0 2 0H11Z",
  ].join("");
}

export function PlayheadNeedle({ height }: { height: number }) {
  const clipId = `playhead-${useId().replace(/[^\w-]/g, "")}`;
  const d = needlePath(height);

  return (
    <svg
      className={styles.needle}
      width={PLAYHEAD_HANDLE_WIDTH_PX}
      height={height}
      viewBox={`0 0 ${PLAYHEAD_HANDLE_WIDTH_PX} ${height}`}
      aria-hidden
    >
      <clipPath id={clipId}>
        <path d={d} />
      </clipPath>
      {/* A 2px stroke clipped to the shape draws a 1px inner outline. */}
      <path d={d} className={styles.needleShape} clipPath={`url(#${clipId})`} />
    </svg>
  );
}
