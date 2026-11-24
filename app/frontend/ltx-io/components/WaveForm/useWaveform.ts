import { useEffect, useState } from "react";

import { ColorScheme } from "@ds/styles/themes/useTheme";
import { handleErrors } from "@ds/lib/handleErrors";
import {
  computeWaveformData,
  decodeAudio,
} from "@/ltx-io/components/WaveForm/waveformUtils";

type UseWaveformArgs = {
  src: string;
  width: number;
  height: number;
  startTime: number;
  /** Defaults to the full audio duration when omitted. */
  endTime?: number;
  highlighted?: boolean;
  colorScheme: ColorScheme;
  pixelRatio?: number;
  styleOverrides?: {
    strokeColor?: string;
    barWidth?: number;
    barGap?: number;
    minBarHeight?: number;
    maxBarHeight?: number;
    lineCap?: CanvasLineCap;
  };
};

const SOUND_WAVEFORM_SAMPLE_WIDTH = 1.75;

// Create a cache map outside the hook to persist across renders and component instances
const canvasCache = new Map<string, string>();

const generateCacheKey = (args: UseWaveformArgs) => {
  const overrides = args.styleOverrides;
  return [
    args.src,
    args.width,
    args.height,
    args.startTime,
    args.endTime,
    args.highlighted ?? "",
    args.colorScheme,
    args.pixelRatio ?? "",
    overrides?.strokeColor ?? "",
    overrides?.barWidth ?? "",
    overrides?.barGap ?? "",
    overrides?.minBarHeight ?? "",
    overrides?.maxBarHeight ?? "",
    overrides?.lineCap ?? "",
  ].join("-");
};

const renderWaveform = (args: {
  waveformData: number[];
  width: number;
  height: number;
  colorScheme: ColorScheme;
  pixelRatio: number;
  styleOverrides?: {
    strokeColor?: string;
    barWidth?: number;
    barGap?: number;
    minBarHeight?: number;
    maxBarHeight?: number;
    lineCap?: CanvasLineCap;
  };
}) => {
  const {
    waveformData,
    width,
    height,
    colorScheme,
    pixelRatio = 1,
    styleOverrides,
  } = args;
  const {
    strokeColor,
    barWidth = 1.25,
    barGap = 0.5,
    minBarHeight = 0.5,
    maxBarHeight = 20,
    lineCap,
  } = styleOverrides ?? {};
  // Create an offscreen canvas
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.floor(width * pixelRatio));
  canvas.height = Math.max(1, Math.floor(height * pixelRatio));
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  ctx.scale(pixelRatio, pixelRatio);

  // Clear the canvas
  ctx.clearRect(0, 0, width, height);

  // Set up styling
  if (strokeColor) {
    ctx.strokeStyle = strokeColor;
  } else {
    switch (colorScheme) {
      case "dark":
        ctx.strokeStyle = "#F3F4F5";
        break;
      case "light":
        ctx.strokeStyle = "#28292C";
        break;
    }
  }
  ctx.lineWidth = barWidth;
  if (lineCap !== undefined) {
    ctx.lineCap = lineCap;
  }
  // Calculate number of bars that can fit in the width with the specified spacing
  const totalWidth = barWidth + barGap;
  if (totalWidth <= 0) return null;
  const totalBars = Math.max(1, Math.floor((width - barWidth) / totalWidth) + 1);

  const halfBar = barWidth / 2;

  for (let i = 0; i < totalBars; i++) {
    const amplitude = waveformData[i];
    const barHeight = minBarHeight + (amplitude / 100) * (maxBarHeight - minBarHeight);

    const x = halfBar + i * totalWidth;
    const y = (height - barHeight) / 2;

    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, y + barHeight);
    ctx.stroke();
  }

  return canvas.toDataURL("image/png");
};

export type WaveformRenderState = "idle" | "loading" | "error";

export function useWaveform(
  args: UseWaveformArgs,
  onStateChange?: (state: WaveformRenderState) => void,
) {
  const cacheKey = generateCacheKey(args);

  const [waveform, setWaveform] = useState<string>();

  useEffect(() => {
    if (canvasCache.has(cacheKey)) {
      const cached = canvasCache.get(cacheKey);
      if (cached) {
        setWaveform(cached);
      }
      return;
    }

    const { src, width, height, startTime, colorScheme, pixelRatio = 1, styleOverrides } = args;
    if (!width || !height) {
      return;
    }

    let cancelled = false;
    onStateChange?.("loading");
    decodeAudio(src)
      .then((buffer) => {
        if (cancelled) {
          return;
        }
        const endTime = args.endTime ?? buffer.duration;
        const numberOfSamples = Math.floor(width / SOUND_WAVEFORM_SAMPLE_WIDTH);
        const waveformData = computeWaveformData(
          buffer,
          startTime,
          endTime - startTime,
          numberOfSamples,
        );
        const rendered = renderWaveform({
          waveformData,
          width,
          height,
          colorScheme,
          pixelRatio,
          styleOverrides,
        });
        if (rendered) {
          canvasCache.set(cacheKey, rendered);
          setWaveform(rendered);
        }
        onStateChange?.("idle");
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        handleErrors([error]);
        onStateChange?.("error");
      });

    return () => {
      cancelled = true;
    };
    // cacheKey is a pure function of all `args` fields, so it fully captures the inputs.
  }, [cacheKey]); // eslint-disable-line react-hooks/exhaustive-deps

  return waveform;
}
