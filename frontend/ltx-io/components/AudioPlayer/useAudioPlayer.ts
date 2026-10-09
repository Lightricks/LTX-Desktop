import { useCallback, useEffect, useRef, useState } from "react";

export type UseAudioPlayerOptions = {
  shouldLoadSrc?: boolean;
  muted?: boolean;
  autoPlay?: boolean;
  loop?: boolean;
  onError?: () => void;
};

export function useAudioPlayer(
  src: string,
  options?: UseAudioPlayerOptions,
): {
  isPlaying: boolean;
  isLoading: boolean;
  play: () => Promise<void>;
  pause: () => void;
  seek: (timeSeconds: number) => void;
  duration: number;
  currentTime: number;
} {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const timeUpdateFrameRef = useRef(0);
  const autoPlayAttemptedRef = useRef(false);
  const onErrorRef = useRef(options?.onError);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const shouldLoad = options?.shouldLoadSrc !== false;
  const muted = options?.muted === true;
  const autoPlay = options?.autoPlay === true;
  const loop = options?.loop === true;
  const mutedRef = useRef(muted);
  const loopRef = useRef(loop);

  onErrorRef.current = options?.onError;
  mutedRef.current = muted;
  loopRef.current = loop;

  useEffect(() => {
    if (!shouldLoad || !src) {
      audioRef.current?.pause();
      audioRef.current = null;
      autoPlayAttemptedRef.current = false;
      setIsPlaying(false);
      setIsLoading(true);
      setDuration(0);
      setCurrentTime(0);
      return;
    }

    const audio = new Audio(src);
    audio.muted = mutedRef.current;
    audio.loop = loopRef.current;
    audioRef.current = audio;
    autoPlayAttemptedRef.current = false;
    setIsLoading(true);
    setIsPlaying(false);
    setDuration(0);
    setCurrentTime(0);

    const handleLoaded = () => {
      setDuration(Number.isFinite(audio.duration) ? audio.duration : 0);
      setIsLoading(false);
    };
    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
    };
    const handleError = () => {
      setIsPlaying(false);
      setIsLoading(false);
      onErrorRef.current?.();
    };

    audio.addEventListener("loadedmetadata", handleLoaded);
    audio.addEventListener("ended", handleEnded);
    audio.addEventListener("error", handleError);
    audio.load();

    return () => {
      audio.pause();
      audio.removeEventListener("loadedmetadata", handleLoaded);
      audio.removeEventListener("ended", handleEnded);
      audio.removeEventListener("error", handleError);
      audioRef.current = null;
    };
  }, [shouldLoad, src]);

  useEffect(() => {
    if (audioRef.current) audioRef.current.muted = muted;
  }, [muted]);

  useEffect(() => {
    if (audioRef.current) audioRef.current.loop = loop;
  }, [loop]);

  // HTMLAudioElement doesn't fire `timeupdate` frequently enough for smooth
  // waveform progress, so we poll currentTime on every animation frame while playing.
  useEffect(() => {
    if (!isPlaying) return;

    const tick = () => {
      const audio = audioRef.current;
      if (audio) {
        setCurrentTime(audio.currentTime);
      }
      timeUpdateFrameRef.current = requestAnimationFrame(tick);
    };
    timeUpdateFrameRef.current = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(timeUpdateFrameRef.current);
  }, [isPlaying]);

  const play = useCallback(async () => {
    const audio = audioRef.current;
    if (!audio) return;
    await audio.play();
    setIsPlaying(true);
  }, []);

  const pause = useCallback(() => {
    audioRef.current?.pause();
    setIsPlaying(false);
  }, []);

  const seek = useCallback((timeSeconds: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    const next = Number.isFinite(timeSeconds) ? Math.max(0, timeSeconds) : 0;
    audio.currentTime = next;
    setCurrentTime(next);
  }, []);

  useEffect(() => {
    if (!autoPlay || isLoading || autoPlayAttemptedRef.current) return;
    autoPlayAttemptedRef.current = true;
    void play().catch(() => undefined);
  }, [autoPlay, isLoading, play]);

  return { isPlaying, isLoading, play, pause, seek, duration, currentTime };
}

export function formatDurationToMMSS(seconds: number): string {
  const total = Number.isFinite(seconds) ? Math.max(0, Math.floor(seconds)) : 0;
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}
