import { useLayoutEffect, useRef, useState, type ReactNode } from "react";

import { Text } from "@/ds/Text/Text";

import { Info } from "../../components/shared/Info/Info.tsx";

import type { components } from "../../../generated/backend-openapi.ts";

import {
  ACTIVITY_SERIES,
  DASHBOARD_INFO,
  failureLabel,
  fixHint,
  formatActivityDay,
  formatFootage,
  formatGpu,
  formatPercent,
  formatSeconds,
} from "./dashboardFormat.ts";
import { RenderPanel } from "./RenderPanel.tsx";
import { SectionTitle } from "./SectionTitle.tsx";
import type { DashboardSelection } from "./useDashboardSelection.ts";
import {
  activityChartHeight,
  dashboardGridStyle,
  dashboardPanelStyle,
  kpiColumns,
  type DashboardLayout,
  type DashboardPanelId,
} from "./dashboardLayout.ts";
import styles from "./DashboardScreen.module.scss";

type Snapshot = components["schemas"]["DashboardSnapshot"];

export function DashboardPanels({
  snapshot,
  width,
  layout,
  selection,
  onSelectionChange,
}: {
  snapshot: Snapshot;
  width: number;
  layout: DashboardLayout;
  selection: DashboardSelection;
  onSelectionChange: (patch: Partial<DashboardSelection>) => void;
}) {
  const panels: Record<DashboardPanelId, ReactNode> = {
    kpis: <KpiPanel snapshot={snapshot} width={width} />,
    activity: <ActivityPanel snapshot={snapshot} width={width} />,
    render: (
      <RenderPanel
        snapshot={snapshot}
        selection={selection}
        onSelectionChange={onSelectionChange}
      />
    ),
    loras: <LoraPanel snapshot={snapshot} />,
    failures: <FailurePanel snapshot={snapshot} />,
    keep: <KeepPanel snapshot={snapshot} />,
  };
  return (
    <div
      className={styles.grid}
      data-columns={layout.columns}
      style={dashboardGridStyle(layout)}
    >
      {layout.order.map((id) => (
        <section
          key={id}
          className={styles.panel}
          data-panel={id}
          style={dashboardPanelStyle(layout, id)}
        >
          {panels[id]}
        </section>
      ))}
    </div>
  );
}

function KpiPanel({ snapshot, width }: { snapshot: Snapshot; width: number }) {
  const kpi = kpiColumns(width);
  const { kpis } = snapshot;
  return (
    <div
      className={styles.kpiGrid}
      style={{ gridTemplateColumns: `repeat(${kpi.columns}, minmax(0, 1fr))` }}
    >
      <article>
        <SectionTitle title="Content created" info={DASHBOARD_INFO.content} />
        <Text as="p" variant="heading" size="xl" className={styles.kpiValue}>
          {kpis.content_count}
        </Text>
        <Text as="p" variant="body" size="md" className={styles.muted}>
          {formatFootage(kpis.footage_s)} of footage
        </Text>
      </article>
      <article>
        <SectionTitle title="GPU time" info={DASHBOARD_INFO.gpu} />
        <Text as="p" variant="heading" size="xl" className={styles.kpiValue}>
          {formatGpu(kpis.gpu_ms)}
        </Text>
        <Text as="p" variant="body" size="md" className={styles.muted}>
          Succeeded and failed runs
        </Text>
      </article>
      <article style={kpi.successSpans ? { gridColumn: "1 / -1" } : undefined}>
        <SectionTitle title="Success rate" info={DASHBOARD_INFO.success} />
        <Text as="p" variant="heading" size="xl" className={styles.kpiValue}>
          {formatPercent(kpis.success_rate)}
        </Text>
        <Text as="p" variant="body" size="md" className={styles.muted}>
          {kpis.succeeded} succeeded, {kpis.failed} failed
          {kpis.cancelled > 0 ? `, ${kpis.cancelled} cancelled` : ""}
        </Text>
      </article>
    </div>
  );
}

function ActivityPanel({ snapshot, width }: { snapshot: Snapshot; width: number }) {
  const max = Math.max(
    1,
    ...snapshot.activity.map((day) =>
      ACTIVITY_SERIES.reduce((sum, series) => sum + (day.counts[series.id] ?? 0), 0),
    ),
  );
  const withYear =
    snapshot.activity.length > 1 &&
    snapshot.activity[0]?.day.slice(0, 4) !==
      snapshot.activity[snapshot.activity.length - 1]?.day.slice(0, 4);
  const scrollRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const node = scrollRef.current;
    if (!node) return;
    node.scrollLeft = node.scrollWidth;
  }, [snapshot.activity.length]);
  return (
    <div>
      <Text as="h2" variant="heading" size="sm" className={styles.panelTitle}>
        Generations
      </Text>
      <div ref={scrollRef} className={styles.chartScroll}>
        <div className={styles.chart}>
          {snapshot.activity.map((day) => {
            const total = ACTIVITY_SERIES.reduce(
              (sum, series) => sum + (day.counts[series.id] ?? 0),
              0,
            );
            const when = formatActivityDay(day.day, withYear);
            const title = activityTitle(day, when);
            return (
              <div
                key={day.day}
                className={styles.bar}
                role="img"
                aria-label={title}
                title={title}
              >
                <div className={styles.plot} style={{ height: activityChartHeight(width) }}>
                  <div className={styles.stack} style={{ height: `${(total / max) * 100}%` }}>
                    {ACTIVITY_SERIES.map((series) => (
                      <span
                        key={series.id}
                        className={styles.segment}
                        data-series={series.id}
                        style={{ flexGrow: day.counts[series.id] ?? 0 }}
                      />
                    ))}
                  </div>
                </div>
                <div className={styles.daySlot}>
                  <Text as="span" variant="body" size="md" className={styles.dayLabel}>
                    {when}
                  </Text>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <div className={styles.legend}>
        {ACTIVITY_SERIES.map((series) => (
          <Text as="span" key={series.id} variant="body" size="md">
            <span className={styles.swatch} data-series={series.id} />
            {series.label}
          </Text>
        ))}
      </div>
    </div>
  );
}

function activityTitle(day: Snapshot["activity"][number], when: string): string {
  const parts = ACTIVITY_SERIES.flatMap((series) => {
    const count = day.counts[series.id] ?? 0;
    return count > 0 ? [`${count} ${series.label.toLowerCase()}`] : [];
  });
  return parts.length > 0 ? `${when}: ${parts.join(", ")}` : `${when}: no generations`;
}

const LORA_PREVIEW_COUNT = 10;

function LoraPanel({ snapshot }: { snapshot: Snapshot }) {
  const { loras } = snapshot;
  const [expanded, setExpanded] = useState(false);
  const hidden = Math.max(0, loras.top.length - LORA_PREVIEW_COUNT);
  const visible = expanded ? loras.top : loras.top.slice(0, LORA_PREVIEW_COUNT);
  return (
    <div>
      <SectionTitle title="LoRAs" info={DASHBOARD_INFO.loras} />
      <Text as="p" variant="body" size="md" className={styles.muted}>
        {loras.share == null ? "No runs yet" : `${formatPercent(loras.share)} of runs`}
      </Text>
      {loras.top.length === 0 ? (
        <Text as="p" variant="body" size="md" className={styles.muted}>
          No LoRA runs in this range.
        </Text>
      ) : (
        <ol className={`${styles.list} ${styles.scrollList}`}>
          {visible.map((lora) => (
            <li key={lora.feature}>
              <Text as="span" variant="body" size="md">
                {lora.name}
              </Text>
              <Text as="span" variant="body" size="md" className={styles.muted}>
                {" "}
                · {lora.runs} runs · keep {formatPercent(lora.keep_rate)}
              </Text>
            </li>
          ))}
        </ol>
      )}
      {hidden > 0 ? (
        <button
          type="button"
          className={styles.more}
          aria-expanded={expanded}
          onClick={() => setExpanded((open) => !open)}
        >
          <Text as="span" variant="body" size="md">
            {expanded ? "Show less" : `+ ${hidden} more`}
          </Text>
        </button>
      ) : null}
    </div>
  );
}

function FailurePanel({ snapshot }: { snapshot: Snapshot }) {
  return (
    <div>
      <Text as="h2" variant="heading" size="sm" className={styles.panelTitle}>
        Failures
      </Text>
      {snapshot.failures.length === 0 ? (
        <Text as="p" variant="body" size="md" className={styles.muted}>
          No failed runs in this range.
        </Text>
      ) : (
        <ul className={`${styles.list} ${styles.scrollList}`}>
          {snapshot.failures.map((failure) => (
            <li key={failure.error_code}>
              <Text as="span" variant="body" size="md">
                {failureLabel(failure.error_code)} · {failure.count}
              </Text>
              <Text as="div" variant="body" size="md" className={styles.muted}>
                {fixHint(failure.error_code)}
              </Text>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function KeepPanel({ snapshot }: { snapshot: Snapshot }) {
  const { keep_rate: keep, usual } = snapshot;
  return (
    <div>
      <SectionTitle title="Keep rate" info={DASHBOARD_INFO.keep} />
      <Text as="p" variant="heading" size="xl" className={styles.kpiValue}>
        {formatPercent(keep.rate)}
      </Text>
      <Text as="p" variant="body" size="md" className={styles.muted}>
        {keep.kept} of {keep.total} clips kept
      </Text>
      <Text as="p" variant="body" size="md">
        <span className={styles.titleWithInfo}>
          Your usual settings
          <Info content={DASHBOARD_INFO.usual} side="top" align="start" maxWidth={320} />
        </span>
        {usual
          ? `: ${usual.resolution}, ${usual.aspect_ratio}, ${formatSeconds(usual.duration_s)}, ${usual.fps} fps`
          : null}
      </Text>
      {usual ? null : (
        <Text as="p" variant="body" size="md" className={styles.muted}>
          These appear after a few finished videos.
        </Text>
      )}
    </div>
  );
}

