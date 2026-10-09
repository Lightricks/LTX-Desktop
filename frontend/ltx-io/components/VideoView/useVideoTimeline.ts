import { useCallback, useEffect, useState, type RefObject } from "react";

/**
 * Elapsed time, duration, and seeking for a `<video>` driving `VideoControls`.
 * Follows playback with rAF so the progress bar moves smoothly, and listens to
 * `seeked`/`timeupdate` so seeks made elsewhere (a trim playhead, a loop
 * restart) show up while paused.
 */
export function useVideoTimeline(
  videoRef: RefObject<HTMLVideoElement | null>,
  src: string,
) {
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let frame: number | null = null;
    const readTime = () => setCurrentTime(video.currentTime);
    const readDuration = () => {
      const next = video.duration;
      setDuration(Number.isFinite(next) && next > 0 ? next : 0);
    };
    const tick = () => {
      readTime();
      frame = requestAnimationFrame(tick);
    };
    const start = () => {
      if (frame === null) frame = requestAnimationFrame(tick);
    };
    const stop = () => {
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
      readTime();
    };

    readDuration();
    readTime();
    if (!video.paused) start();
    video.addEventListener("loadedmetadata", readDuration);
    video.addEventListener("durationchange", readDuration);
    video.addEventListener("seeked", readTime);
    video.addEventListener("timeupdate", readTime);
    video.addEventListener("play", start);
    video.addEventListener("pause", stop);
    video.addEventListener("ended", stop);
    return () => {
      video.removeEventListener("loadedmetadata", readDuration);
      video.removeEventListener("durationchange", readDuration);
      video.removeEventListener("seeked", readTime);
      video.removeEventListener("timeupdate", readTime);
      video.removeEventListener("play", start);
      video.removeEventListener("pause", stop);
      video.removeEventListener("ended", stop);
      if (frame !== null) cancelAnimationFrame(frame);
    };
  }, [videoRef, src]);

  const seek = useCallback(
    (timeSeconds: number) => {
      const video = videoRef.current;
      if (!video) return;
      const next = Number.isFinite(timeSeconds) ? Math.max(0, timeSeconds) : 0;
      video.currentTime = next;
      setCurrentTime(next);
    },
    [videoRef],
  );

  return {
    currentTime,
    duration,
    seek,
    isDragging,
    onDraggingStarted: useCallback(() => setIsDragging(true), []),
    onDraggingEnded: useCallback(() => setIsDragging(false), []),
  };
}
