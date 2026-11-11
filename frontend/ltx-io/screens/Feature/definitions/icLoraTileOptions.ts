import { preflightCellStatus } from "../../../../lib/budget-risk.ts";
import { blockedOptionFields } from "../../../lib/editPreflight.ts";
import type { FieldOption } from "../types.ts";
import type { IcLoraRecipeContext } from "./icLoraRecipe.ts";
import {
  AUTO_NUMBER,
  AUTO_TILE,
  carryFrameOptions,
  overlapChoices,
  overlapLabel,
  TILE_CHOICES,
  tilePresetLabel,
  windowFrameChoices,
  type IcLoraTileChoice,
  type IcLoraTilePreset,
} from "./icLoraTilingControls.ts";

type TileContext = Pick<IcLoraRecipeContext, "tilePreflight" | "tilingTile">;
type WindowContext = Pick<IcLoraRecipeContext, "tilePreflight">;

type PreflightResponse = NonNullable<IcLoraRecipeContext["tilePreflight"]>;
type VerdictCell = Parameters<typeof preflightCellStatus>[0];

/** The status of the first cell of one list that matches, or clear when there is none yet. */
function cellStatus<TCell extends NonNullable<VerdictCell>>(
  context: WindowContext,
  cellsOf: (response: PreflightResponse) => readonly TCell[] | undefined,
  matches: (cell: TCell) => boolean,
) {
  const response = context.tilePreflight;
  return preflightCellStatus(response == null ? undefined : cellsOf(response)?.find(matches));
}

function tileStatus(preset: IcLoraTilePreset, context: TileContext) {
  return cellStatus(context, (response) => response.cells, (cell) => cell.tileSize === preset);
}

/** True when the budget refuses this tile size. Unknown means allowed: Generate decides. */
export function isTileBlocked(preset: IcLoraTilePreset, context: TileContext): boolean {
  return tileStatus(preset, context).status === "blocked";
}

/** The Auto label. Auto runs the catalog tile, which the catalog may not have sent yet. */
function autoTileLabel(context: TileContext): string {
  const tile = context.tilingTile;
  return tile == null ? "Auto" : `Auto (Standard ${tile.long}\u00d7${tile.short})`;
}

/**
 * The tile sizes. The first option is Auto, which runs the catalog tile and is never disabled.
 * A size the budget refuses is disabled and carries the refusal.
 */
export function tileSizeOptions(context: TileContext): FieldOption<IcLoraTileChoice>[] {
  return [
    { value: AUTO_TILE, label: autoTileLabel(context) },
    ...TILE_CHOICES.map((value) => ({
      value,
      label: tilePresetLabel(value, context.tilingTile),
      ...blockedOptionFields(tileStatus(value, context)),
    })),
  ];
}

/**
 * The tile overlaps. The first option is Auto, which runs the overlap of the catalog entry.
 * An overlap does not change what the budget prices, so no option is ever disabled.
 */
export function overlapOptions(context: WindowContext): FieldOption<number>[] {
  const auto = context.tilePreflight?.defaults.overlapFraction;
  return [
    { value: AUTO_NUMBER, label: auto == null ? "Auto" : `Auto (${overlapLabel(auto)})` },
    ...overlapChoices().map((value) => ({ value, label: overlapLabel(value) })),
  ];
}

function windowStatus(maxWindowFrames: number, context: WindowContext) {
  return cellStatus(
    context,
    (response) => response.maxWindowCells,
    (cell) => cell.maxWindowFrames === maxWindowFrames,
  );
}

/** True when the preflight refuses this chunk length. Unknown means allowed: Generate decides. */
export function isWindowBlocked(maxWindowFrames: number, context: WindowContext): boolean {
  return windowStatus(maxWindowFrames, context).status === "blocked";
}

/** The Auto label. The backend says what Auto runs for this clip and resolution. */
function autoChunkLabel(context: WindowContext): string {
  const auto = context.tilePreflight?.defaults.auto;
  if (auto?.kind === "single-pass") return "Auto (single pass)";
  if (auto?.kind === "windowed") return `Auto (${auto.frames} frames)`;
  return "Auto";
}

/**
 * The maximum chunk lengths. The first option is Auto, which is never disabled. A length the
 * preflight refuses is disabled and carries the reason.
 */
export function windowFrameOptions(context: WindowContext): FieldOption<number>[] {
  return [
    { value: AUTO_NUMBER, label: autoChunkLabel(context) },
    ...windowFrameChoices().map((value) => ({
      value,
      label: `${value} frames`,
      ...blockedOptionFields(windowStatus(value, context)),
    })),
  ];
}

function carryStatus(carryFrames: number, context: WindowContext) {
  return cellStatus(
    context,
    (response) => response.carryCells,
    (cell) => cell.carryFrames === carryFrames,
  );
}

/** The Auto label. The backend says which carry Auto runs for this clip and chunk length. */
function autoCarryLabel(context: WindowContext): string {
  const auto = context.tilePreflight?.defaults.carryFrames;
  return auto == null ? "Auto" : `Auto (${auto} frames)`;
}

/**
 * The carries a recipe offers. The first option is Auto, which is never disabled. A carry the
 * preflight refuses is disabled and carries the reason.
 */
export function carryOptions(context: WindowContext): FieldOption<number>[] {
  return [
    { value: AUTO_NUMBER, label: autoCarryLabel(context) },
    ...carryFrameOptions().map((value) => ({
      value,
      label: `${value} frames`,
      ...blockedOptionFields(carryStatus(value, context)),
    })),
  ];
}
