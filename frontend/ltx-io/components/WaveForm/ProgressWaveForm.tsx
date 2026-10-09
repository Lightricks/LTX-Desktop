import { useEffect, useMemo, useRef, useState } from "react";

import {
  WAVEFORM_PLAYED_COLORS,
  WAVEFORM_UNPLAYED_COLORS,
} from "@/ltx-io/components/AudioPlayer/audioWaveformStyle";
import type { WaveformRenderState } from "@/ltx-io/components/WaveForm/useWaveform";
import {
  computeWaveformData,
  decodeAudio,
  drawWaveformBars,
  getWaveformBarCount,
  setupWaveformCanvas,
} from "@/ltx-io/components/WaveForm/waveformUtils";
import { useTheme } from "@ds/styles/themes/useTheme";

export interface ProgressWaveFormProps {
  src: string;
  width: number;
  height: number;
  startTime: number;
  endTime: number;
  /** 0-1 progress for played portion highlight */
  progress: number;
  onStateChange?: (state: WaveformRenderState) => void;
  pixelRatio?: number;
  className?: string;
  /** Color for the played bars */
  progressColor?: string;
  /** Color for the unplayed bars */
  unplayedColor?: string;
  styleOverrides?: {
    barWidth?: number;
    barGap?: number;
    minBarHeight?: number;
    maxBarHeight?: number;
  };
}

/**
 * Full-width waveform with played/unplayed coloring.
 * All bars fit on screen; the played portion is colored dark, unplayed is gray.
 */
export function ProgressWaveForm({
  src,
  width,
  height,
  startTime,
  endTime,
  progress = 0,
  progressColor,
  unplayedColor,
  pixelRatio = 1,
  className,
  onStateChange,
  styleOverrides,
}: ProgressWaveFormProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { colorScheme } = useTheme();

  const playedColor = progressColor ?? WAVEFORM_PLAYED_COLORS[colorScheme];
  const dimColor = unplayedColor ?? WAVEFORM_UNPLAYED_COLORS[colorScheme];

  const {
    barWidth = 2.5,
    barGap = 2,
    minBarHeight = 2,
    maxBarHeight = 28,
  } = styleOverrides ?? {};

  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [decodeFailed, setDecodeFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setAudioBuffer(null);
    setDecodeFailed(false);
    decodeAudio(src)
      .then((buffer) => {
        if (!cancelled) setAudioBuffer(buffer);
      })
      .catch(() => {
        if (!cancelled) setDecodeFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [src]);

  const barStyle = useMemo(
    () => ({ barWidth, barGap, minBarHeight, maxBarHeight }),
    [barWidth, barGap, minBarHeight, maxBarHeight],
  );
  const totalBars = getWaveformBarCount(width, barStyle);

  const samplesResult = useMemo((): { amplitudes: number[] } | { error: true } | null => {
    if (decodeFailed) return { error: true };
    if (!audioBuffer || !width) return null;

    const duration = endTime - startTime;
    if (duration <= 0) return null;

    return {
      amplitudes: computeWaveformData(audioBuffer, startTime, duration, totalBars),
    };
  }, [audioBuffer, decodeFailed, width, startTime, endTime, totalBars]);

  const allSamples =
    samplesResult && "amplitudes" in samplesResult ? samplesResult.amplitudes : null;

  useEffect(() => {
    if (!samplesResult) return;
    onStateChange?.("error" in samplesResult ? "error" : "idle");
  }, [samplesResult, onStateChange]);

  // Redraw all bars with per-bar played/unplayed coloring whenever progress changes.
  // Draws bars directly so each bar can be independently colored based on playhead position.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !allSamples || !width || !height) return;

    const ctx = setupWaveformCanvas(canvas, width, height, pixelRatio);
    if (!ctx) return;

    const progressFraction = Math.min(Math.max(progress, 0), 1);
    const playheadBar = progressFraction * allSamples.length;

    drawWaveformBars(ctx, allSamples, barStyle, height, (i) =>
      i < playheadBar ? playedColor : dimColor,
    );
  }, [allSamples, width, height, pixelRatio, progress, playedColor, dimColor, barStyle]);

  return <canvas ref={canvasRef} className={className} />;
}
