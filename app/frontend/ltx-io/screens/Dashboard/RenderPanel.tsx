import { useState } from "react";

import { Text } from "@/ds/Text/Text";

import type { components } from "../../../generated/backend-openapi.ts";
import { videoGenerationResolutionLabel } from "../../../lib/video-resolution.ts";

import {
  DASHBOARD_INFO,
  formatDuration,
  formatRun,
  formatSeconds,
  renderBand,
} from "./dashboardFormat.ts";
import { FilterMultiSelect } from "./RenderFilters.tsx";
import {
  RENDER_TABS,
  aspectOptions,
  applyFilters,
  cellForRow,
  cellKey,
  groupAdjacent,
  modelOptions,
  renderCellLabel,
  resolutionOptions,
  rowLabel,
  tabCells,
  uniqueRows,
  type GridCell,
  type RenderRowData,
  type RenderTab,
  type RowGroup,
} from "./renderGrid.ts";
import { SectionTitle } from "./SectionTitle.tsx";
import type { DashboardSelection } from "./useDashboardSelection.ts";
import styles from "./DashboardScreen.module.scss";

type Snapshot = components["schemas"]["DashboardSnapshot"];

export function RenderPanel({
  snapshot,
  selection,
  onSelectionChange: setSelection,
}: {
  snapshot: Snapshot;
  selection: DashboardSelection;
  onSelectionChange: (patch: Partial<DashboardSelection>) => void;
}) {
  const [tab, setTab] = useState<RenderTab>("generate");
  const [selected, setSelected] = useState<string | null>(null);
  const { lengthNoun } = RENDER_TABS.find((item) => item.id === tab) ?? RENDER_TABS[0];
  const fpsFilter = selection.fps.map(Number);
  const tabAll = tabCells(snapshot, tab, []);
  const models = modelOptions(tabAll);
  const resolutions = resolutionOptions(tabAll);
  const aspects = tab === "generate" ? snapshot.aspect_ratios : aspectOptions(tabAll);
  const fpsOptions = snapshot.fps_values.map(String);
  const showFps = tab === "generate";
  const hasRenderFilters =
    selection.models.length > 0 ||
    selection.resolutions.length > 0 ||
    selection.aspectRatios.length > 0 ||
    (showFps && selection.fps.length > 0);
  // A saved value this range lacks still filters, so the chips and the grid agree.
  const cells = applyFilters(tabCells(snapshot, tab, showFps ? fpsFilter : []), {
    models: selection.models,
    resolutions: selection.resolutions,
    aspectRatios: selection.aspectRatios,
  });
  const durations = [...new Set(cells.map((cell) => cell.duration_s))].sort((a, b) => a - b);
  const groups = groupAdjacent(uniqueRows(cells), "group");
  const selectedCell = cells.find((cell) => cellKey(cell) === selected) ?? null;
  const columnCount = durations.length + 1;
  return (
    <div>
      <SectionTitle title="Render time" info={DASHBOARD_INFO.render} />
      <div className={`${styles.controls} ${styles.renderTabs}`}>
        {RENDER_TABS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={styles.control}
            aria-pressed={tab === item.id}
            onClick={() => {
              setTab(item.id);
              setSelected(null);
            }}
          >
            <Text as="span" variant="label" size="md">
              {item.label}
            </Text>
          </button>
        ))}
      </div>
      {tab === "generate" ? null : (
        <Text as="p" variant="body" size="md" className={`${styles.muted} ${styles.renderNote}`}>
          {tab === "edit" ? DASHBOARD_INFO.renderEdit : DASHBOARD_INFO.renderIcLora}
        </Text>
      )}
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
          formatOption={videoGenerationResolutionLabel}
          selected={selection.resolutions}
          onChange={(resolutions) => setSelection({ resolutions })}
        />
        <FilterMultiSelect
          label="Aspect ratio"
          options={aspects}
          selected={selection.aspectRatios}
          onChange={(aspectRatios) => setSelection({ aspectRatios })}
        />
        {showFps ? (
          <FilterMultiSelect
            label="FPS"
            options={fpsOptions}
            selected={selection.fps}
            onChange={(fps) => setSelection({ fps })}
          />
        ) : null}
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
          {tabAll.length === 0
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
            <SeriesGroup
              key={group.id}
              group={group}
              durations={durations}
              cells={cells}
              lengthNoun={lengthNoun}
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
          {selectedCell.group ? `${selectedCell.group} · ` : ""}
          {videoGenerationResolutionLabel(selectedCell.resolution)}
          {selectedCell.aspect_ratio ? ` · ${selectedCell.aspect_ratio}` : ""}
          {selectedCell.fps == null ? "" : ` · ${selectedCell.fps} fps`} ·{" "}
          {formatSeconds(selectedCell.duration_s)} {lengthNoun} · median{" "}
          {formatDuration(selectedCell.median_ms)} · 90th percentile{" "}
          {formatDuration(selectedCell.p90_ms)} · {selectedCell.count} runs
        </Text>
      ) : null}
    </div>
  );
}

function SeriesGroup({
  group,
  durations,
  cells,
  lengthNoun,
  selected,
  onSelect,
}: {
  group: RowGroup;
  durations: number[];
  cells: GridCell[];
  lengthNoun: string;
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
          lengthNoun={lengthNoun}
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
  lengthNoun,
  selected,
  onSelect,
}: {
  block: RowGroup;
  durations: number[];
  cells: GridCell[];
  lengthNoun: string;
  selected: string | null;
  onSelect: (key: string) => void;
}) {
  return (
    <>
      <Text as="p" variant="label" size="md" className={styles.renderResolution}>
        {videoGenerationResolutionLabel(block.id)}
      </Text>
      {block.rows.map((row) => (
        <RenderRow
          key={`${row.aspectRatio}:${row.fps ?? ""}`}
          row={row}
          durations={durations}
          cells={cells}
          lengthNoun={lengthNoun}
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
  lengthNoun,
  selected,
  onSelect,
}: {
  row: RenderRowData;
  durations: number[];
  cells: GridCell[];
  lengthNoun: string;
  selected: string | null;
  onSelect: (key: string) => void;
}) {
  return (
    <>
      <Text as="span" variant="body" size="md">
        {rowLabel(row)}
      </Text>
      {durations.map((duration) => {
        const cell = cellForRow(cells, row, duration);
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
            aria-label={renderCellLabel(row, duration, cell.median_ms, lengthNoun)}
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
