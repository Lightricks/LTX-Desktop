import {
  type KeyboardEvent,
  type MouseEvent,
  type RefObject,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  VideoDisplay,
  type VideoDisplayRef,
} from "@ds/VideoDisplay/VideoDisplay";

import { useFeatureFormMuteStore } from "../../stores/featureFormMuteStore";
import {
  VIDEO_PLAYER_CONTROLS,
  VideoControls,
} from "./VideoControls";
import styles from "./VideoPlayer.module.scss";
import {
  seekByNativeVideoStep,
  shouldIgnorePlayerToggle,
  videoPlayerKeyAction,
} from "./videoPlayerKeys";

const CLICK_ACTION_DELAY_MS = 250;

export function VideoPlayer({
  src,
  objectFit = "contain",
  posterUrl,
  label = "Video",
  autoPlay = false,
  fullscreenTargetRef,
}: {
  src: string;
  objectFit?: "contain" | "cover";
  posterUrl?: string;
  label?: string;
  autoPlay?: boolean;
  /**
   * The element that goes fullscreen instead of the player frame. A parent that
   * paints behind the video (the AlphaGen cutout backdrop) passes itself, so its
   * background and controls stay in fullscreen.
   */
  fullscreenTargetRef?: RefObject<HTMLElement | null>;
}) {
  const frameRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<VideoDisplayRef>(null);
  const playingRef = useRef(false);
  const pendingClickRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const clickActedRef = useRef(false);
  const [playing, setPlaying] = useState(false);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const currentTimeRef = useRef(0);
  const durationRef = useRef(0);
  currentTimeRef.current = currentTime;
  durationRef.current = duration;
  const isMuted = useFeatureFormMuteStore((state) => state.isMuted);
  const setMuted = useFeatureFormMuteStore((state) => state.setMuted);
  const fit = isFullScreen ? "contain" : objectFit;

  useEffect(() => {
    playingRef.current = false;
    setPlaying(false);
    setCurrentTime(0);
    setDuration(0);
  }, [src]);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullScreen(
        document.fullscreenElement === (fullscreenTargetRef?.current ?? frameRef.current),
      );
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, [fullscreenTargetRef]);

  useEffect(() => {
    return () => {
      if (pendingClickRef.current != null) clearTimeout(pendingClickRef.current);
    };
  }, []);

  const playOrPause = useCallback(() => {
    if (playingRef.current) {
      playingRef.current = false;
      videoRef.current?.pauseVideo();
      return;
    }
    playingRef.current = true;
    void videoRef.current?.playVideo();
  }, []);

  const togglePlaying = useCallback(
    (event: MouseEvent<Element>) => {
      event.stopPropagation();
      playOrPause();
    },
    [playOrPause],
  );

  const toggleFullscreen = useCallback(() => {
    const frame = fullscreenTargetRef?.current ?? frameRef.current;
    if (!frame) return;
    if (document.fullscreenElement === frame) {
      void document.exitFullscreen().catch(() => {});
      return;
    }
    void frame.requestFullscreen().catch(() => {});
  }, [fullscreenTargetRef]);

  const handleVideoClick = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      if (event.detail > 1) return;
      if (pendingClickRef.current != null) clearTimeout(pendingClickRef.current);
      clickActedRef.current = false;
      pendingClickRef.current = setTimeout(() => {
        pendingClickRef.current = null;
        clickActedRef.current = true;
        playOrPause();
      }, CLICK_ACTION_DELAY_MS);
    },
    [playOrPause],
  );

  const handleVideoDoubleClick = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      event.preventDefault();
      if (pendingClickRef.current != null) {
        clearTimeout(pendingClickRef.current);
        pendingClickRef.current = null;
      } else if (clickActedRef.current) {
        playOrPause();
      }
      clickActedRef.current = false;
      toggleFullscreen();
    },
    [playOrPause, toggleFullscreen],
  );

  const seek = useCallback((timeSeconds: number) => {
    const next = Number.isFinite(timeSeconds) ? Math.max(0, timeSeconds) : 0;
    currentTimeRef.current = next;
    videoRef.current?.seekTo(next);
    setCurrentTime(next);
  }, []);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      const action = videoPlayerKeyAction(event.key);
      if (action == null) return;
      const targetIsButton =
        event.target instanceof HTMLElement && event.target.closest("button") != null;
      // A focused control button already uses Space as its click.
      if (shouldIgnorePlayerToggle(action, targetIsButton)) return;
      event.preventDefault();
      if (action === "toggle-playback") {
        playOrPause();
        return;
      }
      seek(
        seekByNativeVideoStep(
          currentTimeRef.current,
          durationRef.current,
          action === "seek-backward" ? -1 : 1,
        ),
      );
    },
    [playOrPause, seek],
  );

  return (
    <div
      ref={frameRef}
      className={styles.frame}
      tabIndex={0}
      aria-label={label}
      onClick={handleVideoClick}
      onDoubleClick={handleVideoDoubleClick}
      onKeyDown={handleKeyDown}
    >
      <VideoDisplay
        ref={videoRef}
        videoUrl={src}
        fallbackImageUrl={posterUrl}
        preload={autoPlay ? "autoplay" : "metadata"}
        shouldMute={isMuted}
        shouldLoop
        // Native <video controls> ignore pause and fullscreen clicks in Electron.
        shouldShowControls={false}
        objectFit={fit}
        fullWidth
        onLoadMetadata={(event) => {
          const next = event.currentTarget.duration;
          setDuration(Number.isFinite(next) && next > 0 ? next : 0);
        }}
        onTimeChange={setCurrentTime}
        onPlayStart={() => {
          playingRef.current = true;
          setPlaying(true);
        }}
        onPause={() => {
          playingRef.current = false;
          setPlaying(false);
        }}
      />
      <div
        onClick={(event) => event.stopPropagation()}
        onDoubleClick={(event) => event.stopPropagation()}
      >
        <VideoControls
          showElements={VIDEO_PLAYER_CONTROLS}
          currentTime={currentTime}
          duration={duration}
          onSeekCallback={seek}
          onDraggingStarted={() => setIsDragging(true)}
          onDraggingEnded={() => setIsDragging(false)}
          onMute={() => setMuted(!isMuted)}
          isDragging={isDragging}
          isPlaying={playing}
          isMuted={isMuted}
          togglePlaying={togglePlaying}
          onFullScreenClick={toggleFullscreen}
          isFullScreen={isFullScreen}
        />
      </div>
    </div>
  );
}
