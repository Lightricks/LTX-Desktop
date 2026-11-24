import controls from "../../../../../shared/ic-lora-tiling-controls.json" with { type: "json" };

/**
 * The one stored value of Auto in a numeric Advanced control. Each control needs a number for
 * every option, and Auto has none: the backend picks it from the clip and the machine. The form
 * stores Auto as null, and the run never sends this value. A control must not use 0 for it,
 * because a tile overlap of 0 is a real choice.
 */
export const AUTO_NUMBER = -1;

/** The stored value of Auto in the tile size control. */
export const AUTO_TILE = "auto";

/** The tile overlaps a recipe offers, after Auto. The backend reads the same file. */
export function overlapChoices(): readonly number[] {
  return controls.overlap.presets;
}

/** The label of a tile overlap. `0.5` shows as 0.50, so the list lines up. */
export function overlapLabel(value: number): string {
  return value.toFixed(2);
}

export const TILING_OVERLAP_TOOLTIP =
  "Less overlap means fewer tiles and a shorter render. At 4K, 0 uses fewer than half the tiles of 0.50.\n\nA low overlap can leave seams. 0.50 is the safest for quality and the slowest. Auto picks the overlap this LoRA runs well with.";

export const CHUNK_CARRY_TOOLTIP =
  "Each chunk repeats the end of the one before it. A shorter carry means fewer chunks and a shorter render. It can leave a jump where chunks meet.\n\nA longer carry joins more smoothly and takes longer. A clip that fits in one chunk is unchanged.";

/**
 * The carries a recipe offers, after Auto. The preflight says which of them fit the current chunk.
 */
export function carryFrameOptions(): readonly number[] {
  return controls.carryFrames.presets;
}

export const CHUNK_LENGTH_TOOLTIP =
  "The longest chunk a long clip is split into. Auto picks the longest chunk that fits this machine, and runs a short clip in one pass. Auto also picks the carry that lets that chunk run.\n\nA shorter maximum uses less VRAM and takes longer. There are more chunks, and each one repeats the carry. The look is not checked at other lengths.\n\nA chunk must be more than twice the carry. A length that this resolution or machine cannot run is disabled.";

/**
 * The maximum chunk lengths a recipe offers after Auto, longest first. The preflight says which
 * of them this resolution and machine can run.
 */
export function windowFrameChoices(): readonly number[] {
  return controls.windowFrames.presets;
}

/** The tile sizes the preflight judges. Standard is the catalog tile, which Auto runs. */
export type IcLoraTilePreset = "standard" | "large" | "extra-large";

export const TILE_PRESETS: readonly IcLoraTilePreset[] = ["standard", "large", "extra-large"];

/** What the tile size control stores: Auto, or a size that replaces the catalog tile. */
export type IcLoraTileChoice = typeof AUTO_TILE | Exclude<IcLoraTilePreset, "standard">;

/** The sizes the control offers after Auto. */
export const TILE_CHOICES: readonly Exclude<IcLoraTileChoice, typeof AUTO_TILE>[] = [
  "large",
  "extra-large",
];

export function isTileChoice(value: unknown): value is IcLoraTileChoice {
  return value === AUTO_TILE || (TILE_CHOICES as readonly unknown[]).includes(value);
}

export const TILE_SIZE_TOOLTIP =
  "A larger tile means fewer tiles and a shorter render. It needs more VRAM, so a size that does not fit is disabled.\n\nThe LoRA is trained on the standard size. A larger tile can change the detail. A frame that fits in one tile is unchanged.";

const TILE_PRESET_NAME: Record<IcLoraTilePreset, string> = {
  standard: "Standard",
  large: "Large",
  "extra-large": "Extra large",
};

/** The pixels of a tile preset. Standard is the catalog tile, which the catalog may not have sent yet. */
export function tilePresetLabel(
  preset: IcLoraTilePreset,
  catalogTile: { long: number; short: number } | null | undefined,
): string {
  const tile = preset === "standard" ? catalogTile : controls.tileSizes[preset];
  const name = TILE_PRESET_NAME[preset];
  return tile == null ? name : `${name} (${tile.long}\u00d7${tile.short})`;
}
