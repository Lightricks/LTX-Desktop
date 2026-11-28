import { IC_LORA_RECIPES } from "../../../lib/ic-lora-recipes.ts";
import type { components } from "../../../generated/backend-openapi.ts";
import { videoGenerationResolutionLabel } from "../../../lib/video-resolution.ts";

import { formatDuration, formatSeconds } from "./dashboardFormat.ts";

type Snapshot = components["schemas"]["DashboardSnapshot"];

export type RenderTab = "generate" | "edit" | "ic-lora";

export const RENDER_TABS: { id: RenderTab; label: string; lengthNoun: string }[] = [
  { id: "generate", label: "Generate", lengthNoun: "long" },
  { id: "edit", label: "Retake and Extend", lengthNoun: "edited" },
  { id: "ic-lora", label: "IC-LoRA", lengthNoun: "clip" },
];

const EDIT_FEATURES: Record<string, string> = { retake: "Retake", extend: "Extend" };

/** One square of the grid. Only Generate cells can carry a frame rate. */
export type GridCell = {
  group: string;
  model: string;
  resolution: string;
  aspect_ratio: string;
  fps: number | null;
  duration_s: number;
  median_ms: number;
  p90_ms: number;
  count: number;
};

export type RenderRowData = {
  group: string;
  resolution: string;
  aspectRatio: string;
  fps: number | null;
};

export type RowGroup = { id: string; rows: RenderRowData[] };

export type GridFilters = {
  models: string[];
  resolutions: string[];
  aspectRatios: string[];
};

function featureGroup(feature: string, model: string): string {
  const title =
    EDIT_FEATURES[feature] ?? IC_LORA_RECIPES.find((recipe) => recipe.id === feature)?.title ?? feature;
  return model ? `${title} · ${model}` : title;
}

/** Cells of one tab before any filter. The FPS filter only picks a Generate source. */
export function tabCells(snapshot: Snapshot, tab: RenderTab, fps: number[]): GridCell[] {
  if (tab === "generate") {
    const source =
      fps.length === 0
        ? snapshot.render.map((cell) => ({ ...cell, fps: null }))
        : snapshot.render_by_fps.filter((cell) => fps.includes(cell.fps));
    return source.map((cell) => ({ ...cell, group: cell.model }));
  }
  return snapshot.render_by_feature
    .filter((cell) => (cell.feature in EDIT_FEATURES) === (tab === "edit"))
    .map((cell) => ({ ...cell, group: featureGroup(cell.feature, cell.model), fps: null }));
}

export function applyFilters(cells: GridCell[], filters: GridFilters): GridCell[] {
  return cells.filter(
    (cell) =>
      chosen(cell.model, filters.models) &&
      chosen(cell.resolution, filters.resolutions) &&
      chosen(cell.aspect_ratio, filters.aspectRatios),
  );
}

function chosen(value: string, selected: string[]): boolean {
  return selected.length === 0 || selected.includes(value);
}

export function modelOptions(cells: GridCell[]): string[] {
  return [...new Set(cells.map((cell) => cell.model).filter(Boolean))].sort();
}

export function resolutionOptions(cells: GridCell[]): string[] {
  return [...new Set(cells.map((cell) => cell.resolution).filter(Boolean))].sort(
    compareResolutions,
  );
}

export function aspectOptions(cells: GridCell[]): string[] {
  return [...new Set(cells.map((cell) => cell.aspect_ratio).filter(Boolean))].sort();
}

export function groupAdjacent(rows: RenderRowData[], key: "group" | "resolution"): RowGroup[] {
  const groups: RowGroup[] = [];
  for (const row of rows) {
    const id = row[key];
    const current = groups.at(-1);
    if (current?.id === id) current.rows.push(row);
    else groups.push({ id, rows: [row] });
  }
  return groups;
}

export function uniqueRows(cells: GridCell[]): RenderRowData[] {
  const splitFps = new Set(cells.map((cell) => cell.fps).filter((fps) => fps != null)).size > 1;
  const seen = new Set<string>();
  const rows: RenderRowData[] = [];
  for (const cell of cells) {
    const fps = splitFps ? cell.fps : null;
    const key = `${cell.group}:${cell.resolution}:${cell.aspect_ratio}:${fps ?? ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    rows.push({
      group: cell.group,
      resolution: cell.resolution,
      aspectRatio: cell.aspect_ratio,
      fps,
    });
  }
  return rows.sort(compareRows);
}

export function cellForRow(
  cells: GridCell[],
  row: RenderRowData,
  duration: number,
): GridCell | undefined {
  return cells.find(
    (cell) =>
      cell.group === row.group &&
      cell.resolution === row.resolution &&
      cell.aspect_ratio === row.aspectRatio &&
      cell.duration_s === duration &&
      (row.fps == null || cell.fps === row.fps),
  );
}

function compareRows(a: RenderRowData, b: RenderRowData): number {
  const group = a.group.localeCompare(b.group);
  if (group !== 0) return group;
  const resolution = compareResolutions(a.resolution, b.resolution);
  if (resolution !== 0) return resolution;
  const aspect = a.aspectRatio.localeCompare(b.aspectRatio);
  if (aspect !== 0) return aspect;
  return (a.fps ?? 0) - (b.fps ?? 0);
}

function compareResolutions(left: string, right: string): number {
  return resolutionRank(left) - resolutionRank(right) || left.localeCompare(right);
}

function resolutionRank(label: string): number {
  const match = /^(\d+)/.exec(label);
  return match ? Number(match[1]) : Number.POSITIVE_INFINITY;
}

export function rowLabel(row: RenderRowData): string {
  const aspect = row.aspectRatio || videoGenerationResolutionLabel(row.resolution);
  return row.fps == null ? aspect : `${aspect} · ${row.fps} fps`;
}

export function cellKey(cell: GridCell): string {
  return `${cell.group}:${cell.resolution}:${cell.aspect_ratio}:${cell.duration_s}:${cell.fps ?? ""}`;
}

export function renderCellLabel(
  row: RenderRowData,
  duration: number,
  medianMs: number,
  lengthNoun: string,
): string {
  const place = [
    videoGenerationResolutionLabel(row.resolution),
    row.aspectRatio,
    row.fps == null ? "" : `${row.fps} fps`,
    row.group,
  ]
    .filter(Boolean)
    .join(" ");
  return `${place}, ${formatSeconds(duration)} ${lengthNoun}, median ${formatDuration(medianMs)}`;
}
