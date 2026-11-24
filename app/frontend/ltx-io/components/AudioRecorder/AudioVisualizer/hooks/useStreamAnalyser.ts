import { useCallback, useEffect, useRef, useState } from "react";

import type { AnalyserConfig, AnalyserState } from "./types";

export function useStreamAnalyser(
  stream: MediaStream | null,
  isActive: boolean,
  config: AnalyserConfig = {},
): AnalyserState {
  const {
    fftSize = 1024,
    minDecibels = -80,
    maxDecibels = -30,
    smoothingTimeConstant = 0.4,
  } = config;

  const contextRef = useRef<AudioContext | null>(null);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);

  useEffect(() => {
    if (!stream) {
      setAnalyser(null);
      return;
    }

    const ctx = new AudioContext();
    const analyserNode = ctx.createAnalyser();

    analyserNode.fftSize = fftSize;
    analyserNode.minDecibels = minDecibels;
    analyserNode.maxDecibels = maxDecibels;
    analyserNode.smoothingTimeConstant = smoothingTimeConstant;

    const source = ctx.createMediaStreamSource(stream);
    source.connect(analyserNode);

    contextRef.current = ctx;
    setAnalyser(analyserNode);

    return () => {
      source.disconnect();
      analyserNode.disconnect();
      if (ctx.state !== "closed") {
        void ctx.close();
      }
    };
  }, [stream, fftSize, minDecibels, maxDecibels, smoothingTimeConstant]);

  useEffect(() => {
    if (isActive && contextRef.current?.state === "suspended") {
      void contextRef.current.resume();
    }
  }, [isActive]);

  const resume = useCallback(async () => {
    if (contextRef.current?.state === "suspended") {
      await contextRef.current.resume();
    }
  }, []);

  return { analyser, isActive, resume };
}
