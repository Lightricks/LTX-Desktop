import { Button } from "@ds/Button/Button";
import MuteIcon from "@ds/assets/Icons/Audio/Mute.svg?react";
import UnmuteIcon from "@ds/assets/Icons/Audio/On.svg?react";
import PauseIcon from "@ds/assets/Icons/Pause.svg?react";
import PlayIcon from "@ds/assets/Icons/Play.svg?react";
import { clsx } from "clsx";
import { Maximize, Minimize } from "lucide-react";
import {
  type MouseEvent,
  type PointerEvent,
  useCallback,
  useEffect,
  useRef,
} from "react";

import styles from "./VideoControls.module.scss";

export type ShowElements =
  | "play"
  | "mute"
  | "fullscreen"
  | "elapsed"
  | "total"
  | "progress";

export const LIGHTBOX_MEDIA_CONTROLS = new Set<ShowElements>([
  "play",
  "mute",
  "elapsed",
  "total",
  "progress",
]);

/** Feature-form video inputs keep their centered play button, so no play here. */
export const INPUT_PREVIEW_CONTROLS = new Set<ShowElements>([
  "mute",
  "elapsed",
  "total",
  "progress",
]);

/** Play, mute, seek, and fullscreen. Same bar ltx.io's VideoPlayer shows. */
export const VIDEO_PLAYER_CONTROLS = new Set<ShowElements>([
  "play",
  "mute",
  "fullscreen",
  "elapsed",
  "total",
  "progress",
]);

export function VideoControls({
  currentTime,
  duration,
  onSeekCallback,
  onDraggingStarted,
  onDraggingEnded,
  onMute,
  isDragging,
  isPlaying,
  isMuted,
  togglePlaying,
  disabled,
  onFullScreenClick,
  isFullScreen,
  showElements,
}: {
  showElements: ReadonlySet<ShowElements>;
  currentTime: number;
  duration: number;
  onSeekCallback: (currentTime: number) => void;
  onFullScreenClick: () => void;
  onDraggingStarted: () => void;
  onDraggingEnded: () => void;
  onMute?: () => void;
  isDragging: boolean;
  isPlaying: boolean;
  isMuted: boolean;
  isFullScreen: boolean;
  togglePlaying: (event: MouseEvent<Element>) => void;
  disabled?: boolean;
}) {
  const timelineRef = useRef<HTMLDivElement>(null);

  const handleTimelineDragStarted = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      const timeline = timelineRef.current;
      if (!timeline) return;
      seekFromPointer(event, duration, onSeekCallback, timeline);
      onDraggingStarted();
    },
    [duration, onDraggingStarted, onSeekCallback],
  );

  const handleTimelineDrag = useCallback(
    (event: globalThis.PointerEvent) => {
      const timeline = timelineRef.current;
      if (!timeline || !isDragging) return;
      seekFromPointer(event, duration, onSeekCallback, timeline);
    },
    [duration, isDragging, onSeekCallback],
  );

  const handleTimelineDragEnded = useCallback(() => {
    onDraggingEnded();
  }, [onDraggingEnded]);

  useEffect(() => {
    if (!isDragging) return;
    window.addEventListener("pointermove", handleTimelineDrag);
    window.addEventListener("pointerup", handleTimelineDragEnded);
    return () => {
      window.removeEventListener("pointermove", handleTimelineDrag);
      window.removeEventListener("pointerup", handleTimelineDragEnded);
    };
  }, [handleTimelineDrag, handleTimelineDragEnded, isDragging]);

  const progressPercentage =
    duration > 0 && Number.isFinite(duration) && Number.isFinite(currentTime)
      ? Math.min(Math.max((currentTime / duration) * 100, 0), 100)
      : 0;

  return (
    <div
      className={clsx(styles.container, disabled && styles.disabled)}
      data-lightbox-controls
    >
      {showElements.has("play") ? (
        <Button
          size="md"
          appearance="white"
          hierarchy="plain"
          isIconOnly
          leftIcon={isPlaying ? <PauseIcon /> : <PlayIcon />}
          aria-label={isPlaying ? "Pause" : "Play"}
          className={styles.play_button}
          onClick={togglePlaying}
          disabled={disabled}
        />
      ) : null}
      {showElements.has("mute") ? (
        <Button
          size="md"
          appearance="white"
          hierarchy="plain"
          isIconOnly
          leftIcon={isMuted ? <MuteIcon /> : <UnmuteIcon />}
          aria-label={isMuted ? "Unmute" : "Mute"}
          className={styles.play_button}
          onClick={onMute}
          disabled={disabled}
        />
      ) : null}
      <div
        className={clsx(styles.timeline_container, disabled && styles.disabled)}
        onPointerDown={handleTimelineDragStarted}
        onClick={(event) => event.stopPropagation()}
      >
        {showElements.has("elapsed") ? (
          <span className={styles.current_time}>{formattedTime(currentTime)}</span>
        ) : null}
        {showElements.has("progress") ? (
          <div className={styles.timeline} ref={timelineRef}>
            <div className={styles.timeline_bar}>
              <div
                className={styles.played}
                style={{
                  transform: `scaleX(${progressPercentage / 100})`,
                }}
              />
            </div>
            <div
              className={clsx(styles.handle_wrapper, isDragging && styles.dragging)}
              style={{ left: `${progressPercentage}%` }}
            >
              <div className={styles.handle} />
            </div>
          </div>
        ) : null}
        {showElements.has("total") ? (
          <span className={styles.duration_time}>{formattedTime(duration)}</span>
        ) : null}
      </div>

      {showElements.has("fullscreen") ? (
        <Button
          size="md"
          appearance="white"
          hierarchy="plain"
          isIconOnly
          leftIcon={isFullScreen ? <Minimize /> : <Maximize />}
          aria-label={isFullScreen ? "Exit fullscreen" : "Enter fullscreen"}
          onClick={onFullScreenClick}
          disabled={disabled}
        />
      ) : null}
    </div>
  );
}

function seekFromPointer(
  event: PointerEvent<HTMLDivElement> | globalThis.PointerEvent,
  duration: number,
  onSeek: (currentTime: number) => void,
  timeline: HTMLDivElement,
): void {
  const box = timeline.getBoundingClientRect();
  const ratio = (event.clientX - box.left) / box.width;
  onSeek(Math.max(Math.min(ratio * duration, duration), 0));
}

function formattedTime(timeSeconds: number): string {
  const total = Number.isFinite(timeSeconds) ? Math.max(0, Math.floor(timeSeconds)) : 0;
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}
