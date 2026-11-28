import clsx from "clsx";
import { RefObject, useEffect, useState } from "react";

import { ActivityBar } from "@ds/ActivityBar/ActivityBar";

import styles from "./VideoDisplay.module.scss";

export function PlaybackProgress(props: {
  videoRef: RefObject<HTMLVideoElement | null>;
  fps: number | undefined;
  isVisible: boolean;
}) {
  const { videoRef, fps, isVisible } = props;

  const [playback, setPlayback] = useState(0);

  useEffect(() => {
    const { current: element } = videoRef;
    if (!element || !fps || !isVisible) {
      return;
    }
    let animationFrameId: number;
    const updateProgress = () => {
      if (element && !element.paused && !element.ended) {
        const totalFrames = Math.floor(element.duration * fps);
        const currentFrame = Math.floor(element.currentTime * fps);
        setPlayback(totalFrames > 0 ? currentFrame / totalFrames : 0);
      }
      animationFrameId = requestAnimationFrame(updateProgress);
    };
    animationFrameId = requestAnimationFrame(updateProgress);
    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [fps, videoRef, isVisible]);

  if (!fps) {
    return undefined;
  }

  return (
    <div
      className={clsx(styles.playbackProgress, {
        [styles.hidden]: !isVisible,
      })}
    >
      <ActivityBar
        progress={playback * 100}
        size="sm"
        className={styles.playbackProgressBar}
      />
    </div>
  );
}
