import type { components } from "../../../../generated/backend-openapi";
import {
  buildAskMeterModel,
  meterTitle,
  type MemoryFitsCell,
  type MemoryFitsModel,
  type MemoryMeterModel,
} from "../../../../lib/memory-meter.ts";
import { videoGenerationResolutionLabel } from "../../../../lib/video-resolution.ts";
import { formatIcLoraSeconds, icLoraMaxInputSeconds } from "./icLoraFps.ts";

export type IcLoraRecipeMemory = components["schemas"]["IcLoraRecipeMemoryResponse"];

function wholeSeconds(frames: number, fps: number): string {
  const seconds = Math.floor(frames / fps);
  return seconds < 1 ? "under 1 s" : `up to ${seconds} s`;
}

/**
 * Before a clip is picked: each offered cell, filled when a clip at the input cap fits, "up to
 * N s" when only a shorter one does, empty when none does. The line speaks for the selected cell.
 */
export function icLoraFitsModel(
  memory: IcLoraRecipeMemory,
  selected: string,
  fps: number,
): MemoryFitsModel | null {
  if (memory.cells.length === 0 || memory.memory.capacity_gib == null) return null;
  const mac = memory.memory.device === "mac";
  const cells: MemoryFitsCell[] = memory.cells.map((cell) => ({
    value: cell.resolution,
    label: videoGenerationResolutionLabel(cell.resolution),
    state: cell.full ? "full" : cell.max_frames > 0 ? "partial" : "none",
    note: !cell.full && cell.max_frames > 0 ? wholeSeconds(cell.max_frames, fps) : null,
    selected: cell.resolution === selected,
  }));
  const current = cells.find((cell) => cell.selected) ?? null;
  const fit = memory.cells.find((cell) => cell.resolution === selected);
  let line: MemoryFitsModel["line"];
  if (current == null || fit == null) {
    line = { text: "Add a video to see how much memory it needs.", tone: "neutral" };
  } else if (current.state === "full") {
    const cap = formatIcLoraSeconds(icLoraMaxInputSeconds(fps));
    line = { text: `Fits clips up to ${cap} s at ${current.label}.`, tone: "neutral" };
  } else if (current.state === "partial") {
    line = {
      text: `At ${current.label}, clips ${wholeSeconds(fit.max_frames, fps)} fit on this ${mac ? "Mac" : "GPU"}.`,
      tone: "neutral",
    };
  } else {
    line = {
      text: mac
        ? `${current.label} needs more memory than your Mac lets video use.`
        : `${current.label} needs more memory than this GPU has.`,
      tone: "over",
    };
  }
  return { title: meterTitle(mac), cells, line };
}

/** After a clip is picked: the selected cell's real ask for that clip. */
export function icLoraMeterModel(memory: IcLoraRecipeMemory): MemoryMeterModel | null {
  return memory.selected == null ? null : buildAskMeterModel(memory.selected, memory.memory);
}
