/** Short edge from which a low performance machine runs a generation slowly. */
const LOW_PERFORMANCE_MIN_SHORT_EDGE = 720;

export const LOW_PERFORMANCE_RESOLUTION_WARNING =
  "This resolution is likely to take a long time on your Mac. Use 540p or lower for a quicker result.";

/** The warning for a resolution with this short edge, or none on a normal machine. */
export function lowPerformanceResolutionWarning(
  shortEdge: number,
  lowPerformanceMachine: boolean | undefined,
): string | undefined {
  return lowPerformanceMachine === true && shortEdge >= LOW_PERFORMANCE_MIN_SHORT_EDGE
    ? LOW_PERFORMANCE_RESOLUTION_WARNING
    : undefined;
}

export const HIGH_RES_RUN_WARNING =
  "4K runs in sections. It takes minutes on a high-end GPU and hours on a Mac.";

export const HIGH_RES_1440P_RUN_WARNING =
  "1440p takes noticeably longer than 1080p, and long clips take much longer still.";

/**
 * The notice for a text, image or audio to video cell the backend runs in overlapping windows, at
 * any resolution. Nothing anchors a later window to the first one except a short tail of the
 * window before it, so beyond the extra time the result can be less consistent than a single pass.
 * An IC-LoRA recipe has a guide video that anchors every window, so it needs its own, shorter line.
 */
export function chunkedRunNotice(windows: number): string {
  return `This runs in ${windows} sections. It takes longer than generating all at once, and the result may be less consistent: lighting, background or subjects can drift between sections.`;
}

export const HIGH_RES_EDIT_4K_RUN_WARNING =
  "4K takes much longer than 1080p and needs a large GPU.";

/** The warning for a 1440p or 4K generation cell, or none for any other cell. */
export function highResRunWarning(resolution: string): string | undefined {
  if (resolution === "2160p") return HIGH_RES_RUN_WARNING;
  return resolution === "1440p" ? HIGH_RES_1440P_RUN_WARNING : undefined;
}

// Edit sizes sit on the 32 px grid, so a 4K or 1440p source can be a few pixels short.
const EDIT_4K_MIN_SHORT_EDGE = 2128;
const EDIT_1440P_MIN_SHORT_EDGE = 1408;

/** The warning for a Retake or Extend size with this short edge. Edits do not run in windows. */
export function highResEditWarning(shortEdge: number): string | undefined {
  if (shortEdge >= EDIT_4K_MIN_SHORT_EDGE) return HIGH_RES_EDIT_4K_RUN_WARNING;
  return shortEdge >= EDIT_1440P_MIN_SHORT_EDGE ? HIGH_RES_1440P_RUN_WARNING : undefined;
}

/**
 * The warning for a generation cell on this machine: the high-resolution run warning first,
 * because it is the specific one (windows, hours on a Mac). Mac slowness is the fallback.
 *
 * `chunk` is what the backend says about this exact cell. An object means it runs in that many
 * windows, so the notice says so at any resolution; 4K shows that notice instead of its generic
 * speed warning, and 1440p keeps its own hint after it. `null` means it runs in one pass, so no
 * window claim is made. `undefined` means the cell is not known (an automatic duration, or a
 * picker without per-cell data): the resolution-only warning stays.
 */
export function generationResolutionWarning(
  resolution: string,
  lowPerformanceMachine: boolean | undefined,
  chunk?: { windows: number } | null,
): string | undefined {
  if (chunk === undefined) {
    return (
      highResRunWarning(resolution) ??
      lowPerformanceResolutionWarning(Number.parseInt(resolution, 10), lowPerformanceMachine)
    );
  }
  const speedHint = resolution === "1440p" ? HIGH_RES_1440P_RUN_WARNING : undefined;
  const parts = [chunk === null ? undefined : chunkedRunNotice(chunk.windows), speedHint].filter(
    (part): part is string => part !== undefined,
  );
  if (parts.length > 0) return parts.join(" ");
  return lowPerformanceResolutionWarning(Number.parseInt(resolution, 10), lowPerformanceMachine);
}
