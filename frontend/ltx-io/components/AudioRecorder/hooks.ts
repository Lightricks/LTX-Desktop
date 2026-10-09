import { useCallback, useEffect, useRef, useState } from "react";

export function recordingStopOutcome(
  aborting: boolean,
  chunkCount: number,
): "abort" | "empty" | "complete" {
  if (aborting) {
    return "abort";
  }
  if (chunkCount <= 0) {
    return "empty";
  }
  return "complete";
}

export function useAudioRecording({
  stream,
  maxDurationSeconds,
  onRecordingComplete,
  onRecordingFailed,
  onAutoStop,
}: {
  stream: MediaStream | null;
  maxDurationSeconds: number;
  onRecordingComplete: (file: File) => void;
  onRecordingFailed?: () => void;
  onAutoStop?: () => void;
}): {
  isRecording: boolean;
  elapsedSeconds: number;
  startRecording: () => boolean;
  stopRecording: () => void;
  abortRecording: () => void;
} {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isAbortingRef = useRef(false);
  const [isRecording, setIsRecording] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const clearTimer = useCallback(() => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
  }, []);

  const stopRecording = useCallback(() => {
    clearTimer();
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  }, [clearTimer]);

  const abortRecording = useCallback(() => {
    isAbortingRef.current = true;
    clearTimer();
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      mediaRecorderRef.current.stop();
    }
    mediaRecorderRef.current = null;
    audioChunksRef.current = [];
    setElapsedSeconds(0);
    setIsRecording(false);
  }, [clearTimer]);

  const startRecording = useCallback(() => {
    if (!stream || stream.getTracks().length === 0) {
      return false;
    }
    // Strict Mode remounts effects; do not construct a second live recorder.
    if (mediaRecorderRef.current?.state === "recording") {
      return true;
    }
    clearTimer();

    isAbortingRef.current = false;
    audioChunksRef.current = [];
    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : undefined;

    try {
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0 && !isAbortingRef.current) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const outcome = recordingStopOutcome(
          isAbortingRef.current,
          audioChunksRef.current.length,
        );
        if (outcome === "abort") {
          return;
        }
        if (outcome === "empty") {
          onRecordingFailed?.();
          return;
        }
        const type = recorder.mimeType || "audio/webm";
        const file = new File(
          audioChunksRef.current,
          `recording-${Date.now()}.webm`,
          {
            type,
          },
        );
        onRecordingComplete(file);
      };

      recorder.start();
    } catch {
      abortRecording();
      return false;
    }

    setIsRecording(true);
    setElapsedSeconds(0);

    timerIntervalRef.current = setInterval(() => {
      setElapsedSeconds((prev) => {
        const next = prev + 1;
        if (next >= maxDurationSeconds) {
          stopRecording();
          onAutoStop?.();
          return maxDurationSeconds;
        }
        return next;
      });
    }, 1000);
    return true;
  }, [
    abortRecording,
    clearTimer,
    maxDurationSeconds,
    onAutoStop,
    onRecordingComplete,
    onRecordingFailed,
    stopRecording,
    stream,
  ]);

  useEffect(() => {
    return abortRecording;
  }, [abortRecording]);

  return {
    isRecording,
    elapsedSeconds,
    startRecording,
    stopRecording,
    abortRecording,
  };
}

export function useCountdown({
  initialSeconds,
  onComplete,
}: {
  initialSeconds: number;
  onComplete: () => void;
}): {
  secondsRemaining: number;
  startCountdown: () => void;
  abortCountdown: () => void;
} {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const remainingRef = useRef(initialSeconds);
  const completedRef = useRef(false);
  const [secondsRemaining, setSecondsRemaining] = useState(initialSeconds);

  const abortCountdown = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const startCountdown = useCallback(() => {
    abortCountdown();
    completedRef.current = false;
    remainingRef.current = initialSeconds;
    setSecondsRemaining(initialSeconds);
    intervalRef.current = setInterval(() => {
      const next = remainingRef.current - 1;
      remainingRef.current = next;
      if (next <= 0) {
        abortCountdown();
        if (!completedRef.current) {
          completedRef.current = true;
          onComplete();
        }
        return;
      }
      setSecondsRemaining(next);
    }, 1000);
  }, [abortCountdown, initialSeconds, onComplete]);

  useEffect(() => {
    return abortCountdown;
  }, [abortCountdown]);

  return { secondsRemaining, startCountdown, abortCountdown };
}

export function finitePlaybackDuration(
  mediaDuration: number,
  fallbackDuration: number,
): number {
  if (Number.isFinite(mediaDuration) && mediaDuration > 0) {
    return mediaDuration;
  }
  if (Number.isFinite(fallbackDuration) && fallbackDuration > 0) {
    return fallbackDuration;
  }
  return 0;
}

export function playbackProgress(
  currentTime: number,
  mediaDuration: number,
  fallbackDuration: number,
): number {
  const duration = finitePlaybackDuration(mediaDuration, fallbackDuration);
  if (duration <= 0 || !Number.isFinite(currentTime) || currentTime < 0) {
    return 0;
  }
  return Math.min(1, currentTime / duration);
}

export function clampPlaybackOffset(
  offsetSeconds: number,
  durationSeconds: number,
): number {
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
    return 0;
  }
  if (!Number.isFinite(offsetSeconds) || offsetSeconds < 0) {
    return 0;
  }
  return Math.min(offsetSeconds, durationSeconds);
}

type PlaybackGraph = {
  ctx: AudioContext;
  source: AudioBufferSourceNode;
  analyser: AnalyserNode;
  startedAt: number;
  offset: number;
};

function configurePlaybackAnalyser(analyser: AnalyserNode): void {
  analyser.fftSize = 1024;
  analyser.minDecibels = -80;
  analyser.maxDecibels = -30;
  analyser.smoothingTimeConstant = 0.7;
}

function closePlaybackGraph(graph: PlaybackGraph | null): void {
  if (graph == null) {
    return;
  }
  graph.source.onended = null;
  try {
    graph.source.stop();
  } catch {
    // already stopped
  }
  if (graph.ctx.state !== "closed") {
    void graph.ctx.close();
  }
}

function playbackElapsedSeconds(graph: PlaybackGraph): number {
  return graph.offset + (graph.ctx.currentTime - graph.startedAt);
}

export function useAudioPlayback(
  buffer: AudioBuffer | null,
  fallbackDurationSeconds = 0,
): {
  isPlaying: boolean;
  isReady: boolean;
  progress: number;
  durationSeconds: number;
  analyser: AnalyserNode | null;
  togglePlayback: () => void;
  seek: (progress: number) => void;
  reset: () => void;
} {
  const bufferRef = useRef<AudioBuffer | null>(buffer);
  bufferRef.current = buffer;
  const graphRef = useRef<PlaybackGraph | null>(null);
  const offsetRef = useRef(0);
  const timeUpdateFrameRef = useRef(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const durationSeconds =
    buffer != null && Number.isFinite(buffer.duration) && buffer.duration > 0
      ? buffer.duration
      : 0;
  const isReady = durationSeconds > 0;

  const stopGraph = useCallback(() => {
    closePlaybackGraph(graphRef.current);
    graphRef.current = null;
    setAnalyser(null);
  }, []);

  useEffect(() => {
    stopGraph();
    offsetRef.current = 0;
    setIsPlaying(false);
    setProgress(0);
    return () => {
      stopGraph();
    };
  }, [buffer, stopGraph]);

  useEffect(() => {
    if (!isPlaying) return;

    const tick = () => {
      const graph = graphRef.current;
      const buffer = bufferRef.current;
      if (graph != null && buffer != null) {
        setProgress(
          playbackProgress(
            playbackElapsedSeconds(graph),
            buffer.duration,
            fallbackDurationSeconds,
          ),
        );
      }
      timeUpdateFrameRef.current = requestAnimationFrame(tick);
    };
    timeUpdateFrameRef.current = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(timeUpdateFrameRef.current);
  }, [fallbackDurationSeconds, isPlaying]);

  const startGraph = useCallback((offset: number) => {
    const buffer = bufferRef.current;
    if (buffer == null || buffer.duration <= 0) {
      return;
    }
    stopGraph();

    const ctx = new AudioContext();
    const source = ctx.createBufferSource();
    const analyserNode = ctx.createAnalyser();
    configurePlaybackAnalyser(analyserNode);
    source.buffer = buffer;
    source.connect(analyserNode);
    analyserNode.connect(ctx.destination);

    const duration = buffer.duration;
    let startOffset = clampPlaybackOffset(offset, duration);
    if (startOffset >= duration) {
      startOffset = 0;
    }

    source.onended = () => {
      if (graphRef.current?.source !== source) {
        return;
      }
      offsetRef.current = 0;
      graphRef.current = null;
      setAnalyser(null);
      setIsPlaying(false);
      setProgress(0);
      if (ctx.state !== "closed") {
        void ctx.close();
      }
    };

    // Same click as toggle: resume without awaiting so the gesture is not spent.
    void ctx.resume();
    source.start(0, startOffset);
    graphRef.current = {
      ctx,
      source,
      analyser: analyserNode,
      startedAt: ctx.currentTime,
      offset: startOffset,
    };
    offsetRef.current = startOffset;
    setAnalyser(analyserNode);
    setIsPlaying(true);
  }, [stopGraph]);

  const togglePlayback = useCallback(() => {
    if (bufferRef.current == null) {
      return;
    }
    if (isPlaying) {
      const graph = graphRef.current;
      if (graph != null) {
        offsetRef.current = clampPlaybackOffset(
          playbackElapsedSeconds(graph),
          bufferRef.current.duration,
        );
        setProgress(
          playbackProgress(
            offsetRef.current,
            bufferRef.current.duration,
            fallbackDurationSeconds,
          ),
        );
      }
      stopGraph();
      setIsPlaying(false);
      return;
    }
    startGraph(offsetRef.current);
  }, [fallbackDurationSeconds, isPlaying, startGraph, stopGraph]);

  const seek = useCallback(
    (nextProgress: number) => {
      const buffer = bufferRef.current;
      const duration = finitePlaybackDuration(
        buffer?.duration ?? 0,
        fallbackDurationSeconds,
      );
      if (buffer == null || duration <= 0) {
        return;
      }
      const nextOffset = clampPlaybackOffset(nextProgress * duration, duration);
      offsetRef.current = nextOffset;
      setProgress(nextProgress);
      if (isPlaying) {
        startGraph(nextOffset);
      }
    },
    [fallbackDurationSeconds, isPlaying, startGraph],
  );

  const reset = useCallback(() => {
    stopGraph();
    offsetRef.current = 0;
    setIsPlaying(false);
    setProgress(0);
  }, [stopGraph]);

  return {
    isPlaying,
    isReady,
    progress,
    durationSeconds,
    analyser,
    togglePlayback,
    seek,
    reset,
  };
}
