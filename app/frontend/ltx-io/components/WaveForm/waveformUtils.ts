// Decode audio locally with the Web Audio API (replaces remotion's `useAudioData`, dropping
// that dependency and giving us control over decoding + caching). Decoded buffers are cached
// per-src (the promise is cached so concurrent callers share one decode); `decodeAudioData`
// detaches the ArrayBuffer, so we never reuse it.
//
// CSP note: this still `fetch(src)`es the bytes, and production `connect-src` is `'self'` +
// localhost — it does not cover `file:`. When WaveForm is wired up, callers must pass a `blob:`
// URL (or an app-protocol `'self'` URL), not a raw `file:` path, or the fetch is blocked.
let sharedAudioContext: AudioContext | null = null;
const audioBufferCache = new Map<string, Promise<AudioBuffer>>();

function getAudioContext(): AudioContext {
  if (!sharedAudioContext) {
    sharedAudioContext = new AudioContext();
  }
  return sharedAudioContext;
}

export function decodeAudio(src: string): Promise<AudioBuffer> {
  let decoded = audioBufferCache.get(src);
  if (!decoded) {
    decoded = fetch(src)
      .then((response) => response.arrayBuffer())
      .then((buffer) => getAudioContext().decodeAudioData(buffer))
      .catch((error) => {
        // Don't cache a failed decode so a later retry can re-attempt.
        audioBufferCache.delete(src);
        throw error;
      });
    audioBufferCache.set(src, decoded);
  }
  return decoded;
}

/**
 * Drops the decoded PCM for a src. Callers that own a short-lived `blob:` URL must call
 * this when they revoke it: the URL is never reachable again, so the cached buffer (which
 * is far larger than the encoded file) would otherwise be pinned for the session.
 */
export function releaseDecodedAudio(src: string): void {
  audioBufferCache.delete(src);
}

/**
 * RMS amplitude per bucket over the `[startTime, startTime + duration]` window, normalized
 * to 0..100 so it maps onto the same bar-height range remotion's `getWaveformPortion` fed.
 */
export function computeWaveformData(
  buffer: AudioBuffer,
  startTimeInSeconds: number,
  durationInSeconds: number,
  numberOfSamples: number,
): number[] {
  if (numberOfSamples <= 0 || durationInSeconds <= 0) {
    return [];
  }
  const { sampleRate, length } = buffer;
  const channel = buffer.getChannelData(0);
  const startSample = Math.max(0, Math.floor(startTimeInSeconds * sampleRate));
  const endSample = Math.min(
    length,
    Math.floor((startTimeInSeconds + durationInSeconds) * sampleRate),
  );
  const windowLength = endSample - startSample;
  if (windowLength <= 0) {
    return [];
  }

  const rms = new Array<number>(numberOfSamples).fill(0);
  const bucketSize = windowLength / numberOfSamples;
  let peak = 0;
  for (let i = 0; i < numberOfSamples; i++) {
    const bucketStart = startSample + Math.floor(i * bucketSize);
    const bucketEnd = Math.min(endSample, startSample + Math.floor((i + 1) * bucketSize));
    let sumSquares = 0;
    let count = 0;
    for (let j = bucketStart; j < bucketEnd; j++) {
      const value = channel[j];
      sumSquares += value * value;
      count++;
    }
    const amplitude = count > 0 ? Math.sqrt(sumSquares / count) : 0;
    rms[i] = amplitude;
    peak = Math.max(peak, amplitude);
  }

  return peak > 0 ? rms.map((amplitude) => (amplitude / peak) * 100) : rms;
}

/**
 * Returns the device pixel ratio capped at 2, for use as the canvas rendering
 * resolution multiplier.
 *
 * High-DPI (Retina) displays report `window.devicePixelRatio` of 2 or higher.
 * Drawing a canvas at this multiplied resolution and then scaling it down via
 * CSS (`canvas.style.width/height`) produces crisp visuals. Capping at 2
 * avoids excessive memory usage on ultra-high-DPI screens (3x, 4x) where the
 * visual difference is negligible but the canvas buffer size grows quadratically.
 *
 * Falls back to 1 when `window` is not available (e.g. SSR).
 */
export function getWaveformPixelRatio(): number {
  if (typeof window === "undefined") return 1;
  return Math.min(window.devicePixelRatio || 1, 2);
}

/**
 * Prepares a canvas for high-DPI waveform rendering: sizes the buffer,
 * sets the CSS display size, applies the DPR scale transform, and clears.
 *
 * @returns The 2D context ready for drawing at logical coordinates,
 *          or `null` if the context couldn't be obtained.
 */
export function setupWaveformCanvas(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
  dpr: number,
): CanvasRenderingContext2D | null {
  const cw = Math.max(1, Math.floor(width * dpr));
  const ch = Math.max(1, Math.floor(height * dpr));

  canvas.width = cw;
  canvas.height = ch;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return ctx;
}

export interface WaveformBarStyle {
  barWidth: number;
  barGap: number;
  minBarHeight: number;
  maxBarHeight: number;
}

/** Number of bars that fit in the given width for the specified bar style. */
export function getWaveformBarCount(width: number, style: WaveformBarStyle): number {
  const { barWidth, barGap } = style;
  const barStep = barWidth + barGap;
  return Math.max(1, Math.floor((width - barWidth) / barStep) + 1);
}

/**
 * Draws vertically-centered waveform bars onto a 2D canvas context.
 *
 * @param ctx        - Canvas rendering context (already scaled for DPR).
 * @param amplitudes - Normalized 0–100 amplitude per bar.
 * @param style      - Bar dimensions (width, gap, min/max height).
 * @param height     - Logical canvas height (for vertical centering).
 * @param getColor   - Returns the stroke color for a given bar index.
 *                     Use a constant function for single-color waveforms,
 *                     or vary by index for progress-based coloring.
 */
export function drawWaveformBars(
  ctx: CanvasRenderingContext2D,
  amplitudes: number[],
  style: WaveformBarStyle,
  height: number,
  getColor: (barIndex: number) => string,
): void {
  const { barWidth, barGap, minBarHeight, maxBarHeight } = style;
  const barStep = barWidth + barGap;

  ctx.lineWidth = barWidth;
  ctx.lineCap = "round";

  const halfBar = barWidth / 2;

  for (let i = 0; i < amplitudes.length; i++) {
    const barHeight =
      minBarHeight + (amplitudes[i] / 100) * (maxBarHeight - minBarHeight);
    const x = halfBar + i * barStep;
    const y = (height - barHeight) / 2;

    ctx.strokeStyle = getColor(i);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, y + barHeight);
    ctx.stroke();
  }
}
