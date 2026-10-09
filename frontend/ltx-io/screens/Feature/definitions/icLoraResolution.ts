import { lowPerformanceResolutionWarning } from "../../../lib/lowPerformanceWarning.ts";
import type { FieldOption } from "../types.ts";
import type { VideoInputContext } from "./videoFeature.ts";

type SourceSize = Pick<VideoInputContext, "videoWidth" | "videoHeight">;
type OptionsContext = SourceSize & Pick<VideoInputContext, "specs">;

/** Cells the local IC-LoRA envelope can finish, lowest first. */
export const IC_LORA_RESOLUTIONS = ["270p", "360p", "540p", "720p", "1080p"] as const;

export type IcLoraRecipeResolution = (typeof IC_LORA_RESOLUTIONS)[number];

export function isIcLoraResolution(value: unknown): value is IcLoraRecipeResolution {
  return (
    typeof value === "string" &&
    (IC_LORA_RESOLUTIONS as readonly string[]).includes(value)
  );
}

// A 720p or 1080p run needs a source at least that large. The backend never upscales
// a two-stage run, so the form hides the cell for a smaller clip. Unknown size keeps it.
// A recipe that sets `upscalesSource` runs one tiled stage. The backend sizes its canvas
// from the cell, so the source size never limits it and every cell is offered.
// The backend source of truth is backend/runtime_config/ic_lora_local_envelope.py.
// This copy must follow it by hand.
const MIN_SOURCE_SHORT_EDGE: Partial<Record<IcLoraRecipeResolution, number>> = {
  "720p": 704,
  "1080p": 1080,
};

export function icLoraResolutionOffered(
  resolution: IcLoraRecipeResolution,
  context: SourceSize,
  upscalesSource = false,
): boolean {
  if (upscalesSource) return true;
  const minShortEdge = MIN_SOURCE_SHORT_EDGE[resolution];
  const { videoWidth, videoHeight } = context;
  if (minShortEdge == null || videoWidth == null || videoHeight == null) return true;
  return Math.min(videoWidth, videoHeight) >= minShortEdge;
}

// Short edge of each cell. The cell nearest the source, not above it, is "Original",
// as the FPS list does for the rate. The backend source of truth is
// backend/runtime_config/ic_lora_local_envelope.py. This copy must follow it by hand.
export const CELL_SHORT_EDGE: Record<IcLoraRecipeResolution, number> = {
  "1080p": 1080,
  "720p": 720,
  "540p": 540,
  "360p": 360,
  "270p": 270,
};
// Same slack as the 704 minimum for 720p.
const ORIGINAL_TOLERANCE_PX = 16;

/** The cell with the largest short edge. The order of `cells` does not matter. */
function largestCell(
  cells: readonly IcLoraRecipeResolution[],
): IcLoraRecipeResolution | undefined {
  return cells.reduce<IcLoraRecipeResolution | undefined>(
    (best, value) =>
      best == null || CELL_SHORT_EDGE[value] > CELL_SHORT_EDGE[best] ? value : best,
    undefined,
  );
}

function offeredCells(context: SourceSize, upscalesSource: boolean): IcLoraRecipeResolution[] {
  return IC_LORA_RESOLUTIONS.filter((value) =>
    icLoraResolutionOffered(value, context, upscalesSource),
  );
}

/**
 * The cell to run. It is `value` when the source offers it. Otherwise it is
 * `fallback` when offered, or else the largest offered cell.
 */
export function icLoraResolutionFor(
  value: unknown,
  context: SourceSize,
  fallback: IcLoraRecipeResolution,
  upscalesSource = false,
): IcLoraRecipeResolution {
  if (isIcLoraResolution(value) && icLoraResolutionOffered(value, context, upscalesSource)) {
    return value;
  }
  if (icLoraResolutionOffered(fallback, context, upscalesSource)) return fallback;
  return largestCell(offeredCells(context, upscalesSource)) ?? fallback;
}

/** The offered cells, lowest first. Unknown source size tags none. */
export function icLoraResolutionOptions(
  context: OptionsContext,
  upscalesSource = false,
): FieldOption<IcLoraRecipeResolution>[] {
  const offered = offeredCells(context, upscalesSource);
  const { videoWidth, videoHeight } = context;
  const shortEdge =
    videoWidth == null || videoHeight == null ? null : Math.min(videoWidth, videoHeight);
  // The largest offered cell that is not above the source.
  const original =
    shortEdge == null
      ? undefined
      : largestCell(
          offered.filter((value) => CELL_SHORT_EDGE[value] <= shortEdge + ORIGINAL_TOLERANCE_PX),
        );
  return offered.map((value) => {
    const warning = lowPerformanceResolutionWarning(
      CELL_SHORT_EDGE[value],
      context.specs?.low_performance_machine,
    );
    return {
      value,
      label: value === original ? `${value} (Original)` : value,
      ...(warning ? { warning } : {}),
    };
  });
}
