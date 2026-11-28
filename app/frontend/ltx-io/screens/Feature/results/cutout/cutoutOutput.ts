/**
 * A cutout run stores the WebM with alpha first and the matte MP4 second. A run
 * from before the bake stored just the matte, so the first output tells which one
 * this is. Only a cutout recipe asks, so no other feature shows the backdrop strip.
 */
type Outputs = readonly { mime_type: string }[];

const ALPHA_WEBM_MIME = "video/webm";

export function isAlphaWebmMime(mimeType: string): boolean {
  return mimeType === ALPHA_WEBM_MIME;
}

export function hasCutoutOutput(generation: { outputs: Outputs }): boolean {
  const first = generation.outputs[0];
  return first != null && isAlphaWebmMime(first.mime_type);
}

/**
 * WebKit (Safari, and every browser on iOS) plays a VP9 WebM without its alpha
 * channel, so a cutout would show the uncut clip. Chromium and Firefox keep it.
 */
export function supportsAlphaWebm(userAgent: string): boolean {
  if (/iPhone|iPad|iPod/.test(userAgent)) return false;
  return !(/Safari/.test(userAgent) && !/Chrome|Chromium|Android/.test(userAgent));
}

export type CutoutFileTarget<TOutput> = { kind: "alpha" | "matte" | "gif"; output: TOutput };

const GIF_MIME = "image/gif";

/**
 * The files a baked cutout run holds: the WebM with alpha, the matte MP4, then the GIF.
 * A run from before the GIF has no third file. Empty when the run is not a baked
 * cutout, so the caller treats it as one file.
 */
export function cutoutFileTargets<TOutput extends { mime_type: string }>(
  generation: { outputs: readonly TOutput[] },
  isCutoutFeature: boolean,
): CutoutFileTarget<TOutput>[] {
  if (!isCutoutFeature || !hasCutoutOutput(generation)) return [];
  const [alpha, matte, gif] = generation.outputs;
  if (alpha === undefined || matte === undefined) return [];
  const targets: CutoutFileTarget<TOutput>[] = [
    { kind: "alpha", output: alpha },
    { kind: "matte", output: matte },
  ];
  if (gif !== undefined && gif.mime_type === GIF_MIME) {
    targets.push({ kind: "gif", output: gif });
  }
  return targets;
}

/**
 * The output to play and whether it is shown as a cutout. A browser that cannot
 * play alpha gets the matte, the second output, as a plain video. A feature that is
 * not a cutout recipe always plays its first output as a plain video.
 */
export function pickPlayback<TOutput extends { mime_type: string }>(
  generation: { outputs: readonly TOutput[] },
  userAgent: string,
  isCutoutFeature: boolean,
): { output: TOutput | undefined; asCutout: boolean } {
  const [first, matte] = generation.outputs;
  if (!isCutoutFeature || !hasCutoutOutput(generation)) return { output: first, asCutout: false };
  if (supportsAlphaWebm(userAgent)) return { output: first, asCutout: true };
  return { output: matte ?? first, asCutout: false };
}
