interface CustomCanvasRenderingContext2D extends CanvasRenderingContext2D {
  roundRect: (
    x: number,
    y: number,
    w: number,
    h: number,
    radius: number,
  ) => void;
}

/**
 * Calculate bar data from frequency data.
 * @param frequencyData - Raw frequency data from AnalyserNode (0-255 values)
 * @param width - Canvas width in pixels
 * @param barWidth - Width of each bar in pixels
 * @param gap - Gap between bars in pixels
 * @param usableFrequencyRatio - Ratio of frequency bins to use (0-1).
 *   only the lower 40% of frequencies where most audio energy is (voice, music).
 */
export const calculateBarData = (
  frequencyData: Uint8Array,
  width: number,
  barWidth: number,
  gap: number,
  usableFrequencyRatio: number = 0.4,
): number[] => {
  // Only use the lower portion of frequency data where actual audio energy exists
  // For voice/music, most energy is in 0-4kHz range (roughly first 40% of bins)
  const usableLength = Math.floor(frequencyData.length * usableFrequencyRatio);

  let units = width / (barWidth + gap);
  let step = Math.floor(usableLength / units);

  if (units > usableLength) {
    units = usableLength;
    step = 1;
  }

  const data: number[] = [];

  for (let i = 0; i < units; i++) {
    let sum = 0;

    for (let j = 0; j < step && i * step + j < usableLength; j++) {
      sum += frequencyData[i * step + j];
    }
    data.push(sum / step);
  }
  return data;
};

/**
 * Calculate bar data with symmetrical mirroring.
 * The visualization shows frequencies on the left half and mirrors them on the right.
 * @param frequencyData - Raw frequency data from AnalyserNode (0-255 values)
 * @param width - Canvas width in pixels
 * @param barWidth - Width of each bar in pixels
 * @param gap - Gap between bars in pixels
 * @param usableFrequencyRatio - Ratio of frequency bins to use (0-1).
 */
export const calculateBarDataSymmetric = (
  frequencyData: Uint8Array,
  width: number,
  barWidth: number,
  gap: number,
  usableFrequencyRatio: number = 0.4,
): number[] => {
  const totalBars = Math.floor(width / (barWidth + gap));
  const halfBars = Math.floor(totalBars / 2);

  if (halfBars === 0) {
    return [];
  }

  const effectiveHalfWidth = halfBars * (barWidth + gap);

  const halfData = calculateBarData(
    frequencyData,
    effectiveHalfWidth,
    barWidth,
    gap,
    usableFrequencyRatio,
  );

  return [...halfData, ...halfData.slice().reverse()];
};

export const draw = (
  data: number[],
  canvas: HTMLCanvasElement,
  barWidth: number,
  gap: number,
  backgroundColor: string,
  barColor: string,
): void => {
  const canvasHeight = canvas.height;
  const amp = canvasHeight / 2;

  const ctx = canvas.getContext("2d") as CustomCanvasRenderingContext2D;
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvasHeight);

  if (backgroundColor !== "transparent") {
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, canvas.width, canvasHeight);
  }

  data.forEach((dp, i) => {
    ctx.fillStyle = barColor;

    // Scale frequency data (0-255) to canvas height
    // dp/255 gives 0-1, multiply by canvasHeight for full range
    const scaledHeight = (dp / 255) * canvasHeight;

    const x = i * (barWidth + gap);
    const y = amp - scaledHeight / 2;
    const w = barWidth;
    const h = scaledHeight || 1;

    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(x, y, w, h, 50);
      ctx.fill();
    } else {
      ctx.fillRect(x, y, w, h);
    }
  });
};

/**
 * Draw minimal idle bars when audio is not playing.
 * Uses even number of bars for symmetry consistency with the animated visualization.
 */
export const drawIdle = (
  canvas: HTMLCanvasElement,
  barWidth: number,
  gap: number,
  backgroundColor: string,
  barColor: string,
): void => {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (backgroundColor !== "transparent") {
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  const totalBars = Math.floor(canvas.width / (barWidth + gap));
  const halfBars = Math.floor(totalBars / 2);
  const numBars = halfBars * 2;

  const centerY = canvas.height / 2;
  const idleHeight = 2;

  ctx.fillStyle = barColor;
  for (let i = 0; i < numBars; i++) {
    const x = i * (barWidth + gap);
    ctx.fillRect(x, centerY - idleHeight / 2, barWidth, idleHeight);
  }
};
