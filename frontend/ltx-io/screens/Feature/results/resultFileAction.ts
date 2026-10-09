import { cutoutFileTargets } from "./cutout/cutoutOutput.ts";
import {
  CUTOUT_ALPHA_FILE_LABEL,
  CUTOUT_GIF_FILE_LABEL,
  CUTOUT_MATTE_FILE_LABEL,
} from "./resultActionsCopy.ts";

export type ResultFile = { label: string; onSelect: () => void };

/** Reveal where the host has the file on disk. Download where it only has a URL. */
export type ResultFileAction = { kind: "reveal" | "download"; files: ResultFile[] };

type FileOutput = { mime_type: string; name: string; path?: string | null };

const CUTOUT_FILE_LABELS = {
  alpha: CUTOUT_ALPHA_FILE_LABEL,
  matte: CUTOUT_MATTE_FILE_LABEL,
  gif: CUTOUT_GIF_FILE_LABEL,
} as const;

/**
 * The file action of one result. A baked cutout run holds two files, so it lists
 * both with a label each. Any other run lists its one played output.
 * Returns null when no file can be revealed or downloaded.
 */
export function resolveResultFileAction<TOutput extends FileOutput>({
  generation,
  playedOutput,
  isCutoutFeature,
  revealInFolder,
  mediaUrlForAsset,
  download,
}: {
  generation: { outputs: readonly TOutput[] };
  playedOutput: TOutput | undefined;
  isCutoutFeature: boolean;
  revealInFolder: ((path: string) => void) | null;
  mediaUrlForAsset: (output: TOutput) => string | null;
  download: (output: TOutput) => void;
}): ResultFileAction | null {
  const cutout = cutoutFileTargets(generation, isCutoutFeature);
  const candidates =
    cutout.length > 0
      ? cutout.map(({ kind, output }) => ({ label: CUTOUT_FILE_LABELS[kind], output }))
      : playedOutput
        ? [{ label: playedOutput.name, output: playedOutput }]
        : [];

  const files: ResultFile[] = [];
  for (const { label, output } of candidates) {
    const path = output.path;
    if (revealInFolder !== null) {
      if (path != null && path.length > 0) {
        files.push({ label, onSelect: () => revealInFolder(path) });
      }
    } else if (mediaUrlForAsset(output) !== null) {
      files.push({ label, onSelect: () => download(output) });
    }
  }
  if (files.length === 0) return null;
  return { kind: revealInFolder !== null ? "reveal" : "download", files };
}
