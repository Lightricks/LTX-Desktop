import { useEffect, useRef } from "react";

import {
  type WaveformRenderState,
  useWaveform,
} from "@/ltx-io/components/WaveForm/useWaveform.ts";
import { useTheme } from "@ds/styles/themes/useTheme";

interface WaveFormProps {
  src: string;
  width: number;
  height: number;
  startTime: number;
  /** Defaults to the full audio duration when omitted. */
  endTime?: number;
  isHighlighted?: boolean;
  onStateChange?: (state: WaveformRenderState) => void;
  pixelRatio?: number;
  className?: string;
  styleOverrides?: {
    strokeColor?: string;
    barWidth?: number;
    barGap?: number;
    minBarHeight?: number;
    maxBarHeight?: number;
    lineCap?: CanvasLineCap;
  };
}

export function WaveForm(props: WaveFormProps) {
  const { width, height, onStateChange, pixelRatio = 1, className } = props;

  const ref = useRef<HTMLCanvasElement>(null);

  const { colorScheme } = useTheme();

  const render = useWaveform({ ...props, colorScheme, pixelRatio }, onStateChange);

  useEffect(() => {
    if (!render) {
      return;
    }

    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) {
      return;
    }

    canvas.width = Math.max(1, Math.floor(width * pixelRatio));
    canvas.height = Math.max(1, Math.floor(height * pixelRatio));
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.onerror = () => {
      onStateChange?.("error");
    };
    img.src = render;
  }, [width, height, onStateChange, render, colorScheme, pixelRatio]);

  return <canvas ref={ref} width={width} height={height} className={className} />;
}
