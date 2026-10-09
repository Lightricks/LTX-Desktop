import { useState } from "react";

import { Text } from "@/ds/Text/Text";

import type { components } from "../../../generated/backend-openapi.ts";

import {
  DASHBOARD_INFO,
  formatDuration,
  formatRun,
  formatSeconds,
  renderBand,
} from "./dashboardFormat.ts";
import { FilterMultiSelect } from "./RenderFilters.tsx";
import { SectionTitle } from "./SectionTitle.tsx";
import type { DashboardSelection } from "./useDashboardSelection.ts";
import styles from "./DashboardScreen.module.scss";

type Snapshot = components["schemas"]["DashboardSnapshot"];
type RenderCell = components["schemas"]["RenderCell"];
type FpsRenderCell = components["schemas"]["FpsRenderCell"];
type GridCell = RenderCell | FpsRenderCell;

export function RenderPanel({
  snapshot,
  selection,
  onSelectionChange: setSelection,
}: {
  snapshot: Snapshot;
  selection: DashboardSelection;
  onSelectionChange: (patch: Partial<DashboardSelection>) => void;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const models = modelOptions(snapshot.render);
  const resolutions = resolutionOptions(snapshot.render);
  const aspects = snapshot.aspect_ratios;
  const fpsOptions = snapshot.fps_values.map(String);
  const hasRenderFilters =
    selection.models.length > 0 ||
    selection.resolutions.length > 0 ||
    selection.aspectRatios.length > 0 ||
    selection.fps.length > 0;
  // A saved value this range lacks still filters, so the chips and the grid agree.
  const cells = visibleRenderCells(snapshot, {
    models: selection.models,
    resolutions: selection.resolutions,
    aspectRatios: selection.aspectRatios,
    fps: selection.fps.map(Number),
  });
  const durations = [...new Set(cells.map((cell) => cell.duration_s))].sort((a, b) => a - b);
  const groups = groupAdjacent(uniqueRows(cells), "model");
  const selectedCell = cells.find((cell) => cellKey(cell) === selected) ?? null;
  const columnCount = durations.length + 1;
  return (
    <div>
      <SectionTitle title="Render time" info={DASHBOARD_INFO.render} />
      <div className={styles.filterRow}>
        <FilterMultiSelect
          label="Model"
          options={models}
          selected={selection.models}
          onChange={(models) => setSelection({ models })}
        />
        <FilterMultiSelect
          label="Resolution"
          options={resolutions}
          selected={selection.resolutions}
          onChange={(resolutions) => setSelection({ resolutions })}
        />
        <FilterMultiSelect
          label="Aspect ratio"
          options={aspects}
          selected={selection.aspectRatios}
          onChange={(aspectRatios) => setSelection({ aspectRatios })}
        />
        <FilterMultiSelect
          label="FPS"
          options={fpsOptions}
          selected={selection.fps}
          onChange={(fps) => setSelection({ fps })}
        />
        <button
          type="button"
          className={styles.clearFilters}
          disabled={!hasRenderFilters}
          onClick={() =>
            setSelection({ models: [], resolutions: [], aspectRatios: [], fps: [] })
          }
        >
          <Text as="span" variant="body" size="md">
            Clear all
          </Text>
        </button>
      </div>
      {groups.length === 0 ? (
        <Text as="p" variant="body" size="md" className={styles.muted}>
          {snapshot.render.length === 0
            ? "Not enough finished renders in this range yet."
            : "No finished renders for these filters."}
        </Text>
      ) : (
        <div
          className={styles.renderTable}
          style={{ gridTemplateColumns: `repeat(${columnCount}, minmax(0, 1fr))` }}
        >
          <span aria-hidden />
          {durations.map((duration) => (
            <Text
              as="span"
              key={duration}
              variant="label"
              size="md"
              align="center"
              className={`${styles.renderHead} ${styles.renderDuration}`}
            >
              {formatSeconds(duration)}
            </Text>
          ))}
          {groups.map((group) => (
            <ModelGroup
              key={group.id}
              group={group}
              durations={durations}
              cells={cells}
              selected={selected}
              onSelect={setSelected}
            />
          ))}
        </div>
      )}
      <div className={styles.legend}>
        {RENDER_LEGEND.map((item) => (
          <Text as="span" key={item.band} variant="body" size="md">
            <span className={styles.swatch} data-band={item.band}>
              {item.band === "none" ? "–" : null}
            </span>
            {item.label}
          </Text>
        ))}
      </div>
      {selectedCell ? (
        <Text as="p" variant="body" size="md" className={styles.detail}>
          {selectedCell.model ? `${selectedCell.model} · ` : ""}
          {selectedCell.resolution}
          {selectedCell.aspect_ratio ? ` · ${selectedCell.aspect_ratio}` : ""}
          {"fps" in selectedCell ? ` · ${selectedCell.fps} fps` : ""} ·{" "}
          {formatSeconds(selectedCell.duration_s)} long · median{" "}
          {formatDuration(selectedCell.median_ms)} · 90th percentile{" "}
          {formatDuration(selectedCell.p90_ms)} · {selectedCell.count} runs
        </Text>
      ) : null}
    </div>
  );
}

function ModelGroup({
  group,
  durations,
  cells,
  selected,
  onSelect,
}: {
  group: RowGroup;
  durations: number[];
  cells: GridCell[];
  selected: string | null;
  onSelect: (key: string) => void;
}) {
  return (
    <>
      <Text as="p" variant="label" size="md" className={styles.renderModel}>
        {group.id}
      </Text>
      {groupAdjacent(group.rows, "resolution").map((block) => (
        <ResolutionGroup
          key={block.id}
          block={block}
          durations={durations}
          cells={cells}
          selected={selected}
          onSelect={onSelect}
        />
      ))}
    </>
  );
}

function ResolutionGroup({
  block,
  durations,
  cells,
  selected,
  onSelect,
}: {
  block: RowGroup;
  durations: number[];
  cells: GridCell[];
  selected: string | null;
  onSelect: (key: string) => void;
}) {
  return (
    <>
      <Text as="p" variant="label" size="md" className={styles.renderResolution}>
        {block.id}
      </Text>
      {block.rows.map((row) => (
        <RenderRow
          key={`${row.aspectRatio}:${row.fps ?? ""}`}
          row={row}
          durations={durations}
          cells={cells}
          selected={selected}
          onSelect={onSelect}
        />
      ))}
    </>
  );
}

function RenderRow({
  row,
  durations,
  cells,
  selected,
  onSelect,
}: {
  row: RenderRowData;
  durations: number[];
  cells: GridCell[];
  selected: string | null;
  onSelect: (key: string) => void;
}) {
  return (
    <>
      <Text as="span" variant="body" size="md">
        {rowLabel(row)}
      </Text>
      {durations.map((duration) => {
        const cell = cells.find(
          (candidate) =>
            candidate.model === row.model &&
            candidate.resolution === row.resolution &&
            candidate.aspect_ratio === row.aspectRatio &&
            candidate.duration_s === duration &&
            (row.fps == null || ("fps" in candidate && candidate.fps === row.fps)),
        );
        if (!cell) {
          return (
            <span key={duration} className={styles.renderEmpty} title="Not tried">
              –
            </span>
          );
        }
        const key = cellKey(cell);
        return (
          <button
            key={duration}
            type="button"
            className={styles.renderCell}
            data-band={renderBand(cell.median_ms)}
            aria-pressed={selected === key}
            aria-label={renderCellLabel(row, duration, cell.median_ms)}
            title={`Median ${formatDuration(cell.median_ms)}`}
            onClick={() => onSelect(key)}
          >
            <Text as="span" variant="label" size="md">
              {formatRun(cell.median_ms)}
            </Text>
          </button>
        );
      })}
    </>
  );
}

const RENDER_LEGEND = [
  { band: "fast", label: "Under 1 min" },
  { band: "mid", label: "1–3 min" },
  { band: "slow", label: "Over 3 min" },
  { band: "none", label: "Not tried" },
] as const;

type RenderRowData = {
  model: string;
  resolution: string;
  aspectRatio: string;
  fps: number | null;
};

function rowLabel(row: RenderRowData): string {
  const aspect = row.aspectRatio || row.resolution;
  return row.fps == null ? aspect : `${aspect} · ${row.fps} fps`;
}

function modelOptions(cells: RenderCell[]): string[] {
  return [...new Set(cells.map((cell) => cell.model).filter(Boolean))].sort();
}

function resolutionOptions(cells: RenderCell[]): string[] {
  return [...new Set(cells.map((cell) => cell.resolution).filter(Boolean))].sort(
    compareResolutions,
  );
}

type RowGroup = { id: string; rows: RenderRowData[] };

function groupAdjacent(rows: RenderRowData[], key: "model" | "resolution"): RowGroup[] {
  const groups: RowGroup[] = [];
  for (const row of rows) {
    const id = row[key];
    const current = groups.at(-1);
    if (current?.id === id) current.rows.push(row);
    else groups.push({ id, rows: [row] });
  }
  return groups;
}

function uniqueRows(cells: GridCell[]): RenderRowData[] {
  const splitFps = distinctFps(cells).length > 1;
  const seen = new Set<string>();
  const rows: RenderRowData[] = [];
  for (const cell of cells) {
    const fps = splitFps && "fps" in cell ? cell.fps : null;
    const key = `${cell.model}:${cell.resolution}:${cell.aspect_ratio}:${fps ?? ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    rows.push({
      model: cell.model,
      resolution: cell.resolution,
      aspectRatio: cell.aspect_ratio,
      fps,
    });
  }
  return rows.sort(compareRows);
}

function distinctFps(cells: GridCell[]): number[] {
  return [...new Set(cells.flatMap((cell) => ("fps" in cell ? [cell.fps] : [])))];
}

function compareRows(a: RenderRowData, b: RenderRowData): number {
  const model = a.model.localeCompare(b.model);
  if (model !== 0) return model;
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

function renderCellLabel(row: RenderRowData, duration: number, medianMs: number): string {
  const place = [
    row.resolution,
    row.aspectRatio,
    row.fps == null ? "" : `${row.fps} fps`,
    row.model,
  ]
    .filter(Boolean)
    .join(" ");
  return `${place}, ${formatSeconds(duration)} long, median ${formatDuration(medianMs)}`;
}

function cellKey(cell: GridCell): string {
  const fps = "fps" in cell ? cell.fps : "";
  return `${cell.model}:${cell.resolution}:${cell.aspect_ratio}:${cell.duration_s}:${fps}`;
}

function visibleRenderCells(
  snapshot: Snapshot,
  filters: {
    models: string[];
    resolutions: string[];
    aspectRatios: string[];
    fps: number[];
  },
): GridCell[] {
  const source =
    filters.fps.length === 0
      ? snapshot.render
      : snapshot.render_by_fps.filter((cell) => filters.fps.includes(cell.fps));
  return source.filter(
    (cell) =>
      chosen(cell.model, filters.models) &&
      chosen(cell.resolution, filters.resolutions) &&
      chosen(cell.aspect_ratio, filters.aspectRatios),
  );
}

function chosen(value: string, selected: string[]): boolean {
  return selected.length === 0 || selected.includes(value);
}
