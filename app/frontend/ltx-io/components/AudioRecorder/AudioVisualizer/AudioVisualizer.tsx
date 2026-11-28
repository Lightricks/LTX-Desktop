import {
  type ReactElement,
  useCallback,
  useMemo,
  useRef,
} from "react";

import {
  type AnalyserConfig,
  useFrequencyAnimation,
  useStreamAnalyser,
} from "./hooks";
import { calculateBarDataSymmetric, draw, drawIdle } from "./utils";

export interface VisualizerStyleProps {
  width?: number | string;
  height?: number | string;
  barWidth?: number;
  gap?: number;
  backgroundColor?: string;
  barColor?: string;
}

interface StreamProps {
  source: "stream";
  stream: MediaStream | null;
  isActive: boolean;
  analyser?: never;
  isPlaying?: never;
}

interface PlaybackAnalyserProps {
  source: "analyser";
  analyser: AnalyserNode | null;
  isPlaying: boolean;
  stream?: never;
  isActive?: never;
}

export type AudioVisualizerProps = VisualizerStyleProps &
  AnalyserConfig &
  (StreamProps | PlaybackAnalyserProps);

export function AudioVisualizer(props: AudioVisualizerProps): ReactElement {
  const {
    width = 400,
    height = 150,
    barWidth = 2,
    gap = 4,
    backgroundColor = "transparent",
    barColor = "rgb(160, 198, 255)",

    fftSize = 1024,
    maxDecibels = -30,
    minDecibels = -80,
    smoothingTimeConstant = 0.7,
  } = props;

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isStreamSource = props.source === "stream";

  const analyserConfig: AnalyserConfig = useMemo(
    () => ({
      fftSize,
      minDecibels,
      maxDecibels,
      smoothingTimeConstant,
    }),
    [fftSize, minDecibels, maxDecibels, smoothingTimeConstant],
  );

  const streamResult = useStreamAnalyser(
    isStreamSource ? props.stream : null,
    isStreamSource ? props.isActive : false,
    analyserConfig,
  );

  const analyser = isStreamSource ? streamResult.analyser : props.analyser;
  const isActive = isStreamSource ? streamResult.isActive : props.isPlaying;

  const handleFrame = useCallback(
    (frequencyData: Uint8Array) => {
      if (!canvasRef.current) return;
      const canvas = canvasRef.current;
      const dataPoints = calculateBarDataSymmetric(
        frequencyData,
        canvas.width,
        barWidth,
        gap,
      );
      draw(dataPoints, canvas, barWidth, gap, backgroundColor, barColor);
    },
    [barWidth, gap, backgroundColor, barColor],
  );

  const handleIdle = useCallback(() => {
    if (!canvasRef.current) return;
    drawIdle(canvasRef.current, barWidth, gap, backgroundColor, barColor);
  }, [barWidth, gap, backgroundColor, barColor]);

  useFrequencyAnimation(analyser, isActive, {
    onFrame: handleFrame,
    onIdle: handleIdle,
  });

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        aspectRatio: "unset",
        width: "100%",
        height: "100%",
      }}
    />
  );
}
