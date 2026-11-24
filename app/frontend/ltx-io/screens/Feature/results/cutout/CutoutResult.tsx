import { memo, useRef } from "react";

import { VideoPlayer } from "../../../../components/VideoView/VideoPlayer";
import { useCutoutBackgroundStore } from "../../../../stores/cutoutBackgroundStore";

import { CutoutBackgroundStrip } from "./CutoutBackgroundStrip";
import styles from "./CutoutResult.module.scss";

// A color drag re-renders the stage on every pointer move. The player has the same
// props each time, so it skips the render and keeps its listeners.
const StablePlayer = memo(VideoPlayer);

/**
 * The AlphaGen cutout. The WebM has a real alpha channel, so the backdrop is a
 * CSS background behind the player and a color change needs no new frame.
 */
export function CutoutResult({ src }: { src: string }) {
  const stageRef = useRef<HTMLDivElement>(null);
  const background = useCutoutBackgroundStore((state) => state.background);
  const color = useCutoutBackgroundStore((state) => state.color);

  return (
    <div
      ref={stageRef}
      className={background === "checkerboard" ? styles.checkerboardStage : styles.stage}
      style={background === "color" ? { backgroundColor: color } : undefined}
      data-testid="cutout-result"
    >
      <StablePlayer src={src} label="AlphaGen cutout" fullscreenTargetRef={stageRef} />
      <CutoutBackgroundStrip />
    </div>
  );
}
