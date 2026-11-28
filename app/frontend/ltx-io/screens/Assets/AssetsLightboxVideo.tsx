import { clsx } from "clsx";
import { useCallback, useEffect, useRef, useState, type MouseEvent } from "react";

import {
  LIGHTBOX_MEDIA_CONTROLS,
  VideoControls,
} from "@/ltx-io/components/VideoView/VideoControls";
import { useVideoTimeline } from "@/ltx-io/components/VideoView/useVideoTimeline";
import { useFeatureFormMuteStore } from "@/ltx-io/stores/featureFormMuteStore";

import styles from "./AssetsLightbox.module.scss";
import backdropStyles from "./cutoutBackdrop.module.scss";

export function AssetsLightboxVideo({
  src,
  isCutout = false,
  onError,
}: {
  src: string;
  /** A WebM with alpha. It plays over the same transparency grid as its thumbnail. */
  isCutout?: boolean;
  onError: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const isMuted = useFeatureFormMuteStore((state) => state.isMuted);
  const setMuted = useFeatureFormMuteStore((state) => state.setMuted);
  const [isPlaying, setIsPlaying] = useState(false);
  const timeline = useVideoTimeline(videoRef, src);
  const { seek } = timeline;

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleEnded = () => {
      setIsPlaying(false);
      seek(0);
    };
    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);

    video.addEventListener("ended", handleEnded);
    video.addEventListener("play", handlePlay);
    video.addEventListener("pause", handlePause);
    video.addEventListener("error", onError);
    return () => {
      video.removeEventListener("ended", handleEnded);
      video.removeEventListener("play", handlePlay);
      video.removeEventListener("pause", handlePause);
      video.removeEventListener("error", onError);
    };
  }, [onError, seek, src]);

  const togglePlaying = useCallback((event: MouseEvent<Element>) => {
    event.stopPropagation();
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      void video.play();
    } else {
      video.pause();
    }
  }, []);

  return (
    <div
      className={clsx(
        styles.videoPlayer,
        isCutout && backdropStyles.cutoutBackdropLarge,
      )}
    >
      <video
        ref={videoRef}
        className={styles.video}
        src={src}
        autoPlay
        muted={isMuted}
        playsInline
        onClick={togglePlaying}
      />
      <VideoControls
        showElements={LIGHTBOX_MEDIA_CONTROLS}
        currentTime={timeline.currentTime}
        duration={timeline.duration}
        onSeekCallback={seek}
        onDraggingStarted={timeline.onDraggingStarted}
        onDraggingEnded={timeline.onDraggingEnded}
        onMute={() => setMuted(!isMuted)}
        isDragging={timeline.isDragging}
        isPlaying={isPlaying}
        isMuted={isMuted}
        togglePlaying={togglePlaying}
        onFullScreenClick={() => undefined}
        isFullScreen
      />
    </div>
  );
}
