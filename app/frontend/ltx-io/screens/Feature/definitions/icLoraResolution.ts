import stage2Upscale from "../../../../../shared/ic-lora-stage-2-upscale.json" with { type: "json" };
import { videoGenerationResolutionLabel } from "../../../../lib/video-resolution.ts";
import { highResRunWarning, lowPerformanceResolutionWarning } from "../../../lib/lowPerformanceWarning.ts";
import type { FieldOption } from "../types.ts";
import type { VideoInputContext } from "./videoFeature.ts";

type SourceSize = Pick<VideoInputContext, "videoWidth" | "videoHeight"> & {
  /** The cells above the local envelope that the screen offers (the `highResVideoToVideo` flag). */
  highResCells?: readonly HighResCell[];
  /**
   * The recipe keeps the IC-LoRA on stage 2 (the catalog `stage_2_ic_lora`). Its canvas is the
   * stage-2 size, so a clip up to `maxScale` times smaller than a cell can still fill it.
   */
  twoStage?: boolean;
};
type OptionsContext = SourceSize & Pick<VideoInputContext, "specs">;

/** Cells the local IC-LoRA envelope can finish, lowest first. */
export const IC_LORA_RESOLUTIONS = ["270p", "360p", "540p", "720p", "1080p"] as const;

/**
 * Cells above the local envelope. Every recipe runs them as tiles, with the
 * `highResVideoToVideo` flag. See backend/runtime_config/ic_lora_high_res.py.
 */
export const HIGH_RES_CELLS = ["1440p", "2160p"] as const;
type HighResCell = (typeof HIGH_RES_CELLS)[number];

export type IcLoraRecipeResolution =
  | (typeof IC_LORA_RESOLUTIONS)[number]
  | (typeof HIGH_RES_CELLS)[number];

export function isIcLoraResolution(value: unknown): value is IcLoraRecipeResolution {
  return (
    typeof value === "string" &&
    ([...IC_LORA_RESOLUTIONS, ...HIGH_RES_CELLS] as readonly string[]).includes(value)
  );
}

// A 720p or 1080p run needs a source at least that large. The backend never upscales
// a two-stage run, so the form hides the cell for a smaller clip. Unknown size keeps it.
// A recipe that sets `upscalesSource` runs one tiled stage. The backend sizes its canvas
// from the cell, so the source size never limits it. `"source-and-above"` is the exception:
// it lists the cell nearest the source and the cells above it, and nothing below.
// The backend source of truth is backend/runtime_config/ic_lora_local_envelope.py.
// This copy must follow it by hand. A two-stage recipe is the exception: it offers every
// cell up to `maxScale` times the source (shared/ic-lora-stage-2-upscale.json), 1080p at most.
const MIN_SOURCE_SHORT_EDGE: Partial<Record<IcLoraRecipeResolution, number>> = {
  "720p": 704,
  "1080p": 1080,
  "1440p": 1424,
  "2160p": 2144,
};

/**
 * How a recipe treats a source that is smaller than a cell. `false` keeps the source-size
 * limit. `true` offers every cell up to 1080p (Restore). `"source-and-above"` is for a recipe
 * that scales the clip up with Lanczos (Refine Details). It lists the Original cell and every
 * higher cell, 1440p and 2160p included. A cell below the source would shrink the clip, so
 * it is not listed.
 */
export type UpscalesSource = boolean | "source-and-above";

export function icLoraResolutionOffered(
  resolution: IcLoraRecipeResolution,
  context: SourceSize,
  upscalesSource: UpscalesSource = false,
): boolean {
  const highCell = (HIGH_RES_CELLS as readonly string[]).includes(resolution);
  // A two-stage recipe has no tiled cell, whatever the flag lists. The backend refuses one.
  if (context.twoStage === true && highCell) return false;
  if (highCell && !(context.highResCells as readonly string[] | undefined)?.includes(resolution)) {
    return false;
  }
  if (context.twoStage === true) {
    const { videoWidth, videoHeight } = context;
    // The reach only adds cells. A cell the source-size rule offers stays offered, so a tiny
    // source still lists the smallest cell.
    if (
      videoWidth != null &&
      videoHeight != null &&
      CELL_SHORT_EDGE[resolution] <=
        stage2Upscale.maxScale * Math.min(videoWidth, videoHeight) + ORIGINAL_TOLERANCE_PX
    ) {
      return true;
    }
  }
  if (upscalesSource === "source-and-above") {
    const original = originalCell(context);
    return original == null || CELL_SHORT_EDGE[resolution] >= CELL_SHORT_EDGE[original];
  }
  // Only a source-and-above recipe upscales into a high cell. For the rest the canvas is the
  // clip rounded up to the grid, so a clip under the cell is refused above.
  if (upscalesSource && !highCell) return true;
  const minShortEdge = MIN_SOURCE_SHORT_EDGE[resolution];
  const { videoWidth, videoHeight } = context;
  if (minShortEdge == null || videoWidth == null || videoHeight == null) return true;
  return Math.min(videoWidth, videoHeight) >= minShortEdge;
}

// Short edge of each cell. The cell nearest the source, not above it, is "Original",
// as the FPS list does for the rate. The backend source of truth is
// backend/runtime_config/ic_lora_local_envelope.py. This copy must follow it by hand.
export const CELL_SHORT_EDGE: Record<IcLoraRecipeResolution, number> = {
  "2160p": 2160,
  "1440p": 1440,
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

/**
 * The cell nearest the source, not above it, among the cells the screen offers. It is
 * undefined while the source size is unknown or when the source is below every cell.
 */
function originalCell(context: SourceSize): IcLoraRecipeResolution | undefined {
  const { videoWidth, videoHeight } = context;
  if (videoWidth == null || videoHeight == null) return undefined;
  const shortEdge = Math.min(videoWidth, videoHeight);
  const screenCells = [...IC_LORA_RESOLUTIONS, ...(context.highResCells ?? [])];
  return largestCell(
    screenCells.filter((value) => CELL_SHORT_EDGE[value] <= shortEdge + ORIGINAL_TOLERANCE_PX),
  );
}

function offeredCells(context: SourceSize, upscalesSource: UpscalesSource): IcLoraRecipeResolution[] {
  return [...IC_LORA_RESOLUTIONS, ...HIGH_RES_CELLS].filter((value) =>
    icLoraResolutionOffered(value, context, upscalesSource),
  );
}

/** The cell with the smallest short edge. */
function smallestCell(
  cells: readonly IcLoraRecipeResolution[],
): IcLoraRecipeResolution | undefined {
  return cells.reduce<IcLoraRecipeResolution | undefined>(
    (best, value) =>
      best == null || CELL_SHORT_EDGE[value] < CELL_SHORT_EDGE[best] ? value : best,
    undefined,
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
  upscalesSource: UpscalesSource = false,
): IcLoraRecipeResolution {
  if (isIcLoraResolution(value) && icLoraResolutionOffered(value, context, upscalesSource)) {
    return value;
  }
  if (icLoraResolutionOffered(fallback, context, upscalesSource)) return fallback;
  // A recipe that lists no cell below the source starts at the Original cell, not at 4K.
  const pick = upscalesSource === "source-and-above" ? smallestCell : largestCell;
  return pick(offeredCells(context, upscalesSource)) ?? fallback;
}

/** The offered cells, lowest first. Unknown source size tags none. */
export function icLoraResolutionOptions(
  context: OptionsContext,
  upscalesSource: UpscalesSource = false,
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
    const warning =
      highResRunWarning(value) ??
      lowPerformanceResolutionWarning(
        CELL_SHORT_EDGE[value],
        context.specs?.low_performance_machine,
      );
    return {
      value,
      label:
        value === original
          ? `${videoGenerationResolutionLabel(value)} (Original)`
          : videoGenerationResolutionLabel(value),
      ...(warning ? { warning } : {}),
    };
  });
}

/** The high cells a recipe offers. They need `highResVideoToVideo`, unless the recipe is `highResWithoutFlag`. */
export function recipeHighResCells(
  recipe: { readonly highResWithoutFlag?: boolean },
  highResVideoToVideo: boolean,
): readonly HighResCell[] {
  return highResVideoToVideo || recipe.highResWithoutFlag === true ? HIGH_RES_CELLS : [];
}
