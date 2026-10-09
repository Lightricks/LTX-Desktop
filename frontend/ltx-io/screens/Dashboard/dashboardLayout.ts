export const DASHBOARD_PANELS = [
  "kpis",
  "activity",
  "render",
  "loras",
  "failures",
  "keep",
] as const;

export type DashboardPanelId = (typeof DASHBOARD_PANELS)[number];

export const DASHBOARD_TABLET_MIN = 640;
export const DASHBOARD_DESKTOP_MIN = 1040;

const WIDE_ORDER = [
  "kpis",
  "activity",
  "loras",
  "render",
  "failures",
  "keep",
] as const satisfies readonly DashboardPanelId[];

export type DashboardLayout = {
  columns: 1 | 2 | 3;
  order: readonly DashboardPanelId[];
  column: Record<DashboardPanelId, string>;
  row: Partial<Record<DashboardPanelId, string>>;
};

export function dashboardLayout(width: number): DashboardLayout {
  if (width >= DASHBOARD_DESKTOP_MIN) {
    return {
      columns: 3,
      order: WIDE_ORDER,
      column: {
        kpis: "1 / -1",
        activity: "1 / span 2",
        loras: "3",
        render: "1 / span 2",
        failures: "3",
        keep: "3",
      },
      row: {
        activity: "2",
        loras: "2",
        render: "3 / span 2",
        failures: "3",
        keep: "4",
      },
    };
  }
  if (width >= DASHBOARD_TABLET_MIN) {
    return {
      columns: 2,
      order: DASHBOARD_PANELS,
      column: {
        kpis: "1 / -1",
        activity: "1 / -1",
        render: "1 / -1",
        loras: "auto",
        failures: "auto",
        keep: "1 / -1",
      },
      row: {},
    };
  }
  return {
    columns: 1,
    order: DASHBOARD_PANELS,
    column: {
      kpis: "auto",
      activity: "auto",
      render: "auto",
      loras: "auto",
      failures: "auto",
      keep: "auto",
    },
    row: {},
  };
}

export function kpiColumns(width: number): { columns: 2 | 3; successSpans: boolean } {
  if (width >= DASHBOARD_TABLET_MIN) return { columns: 3, successSpans: false };
  return { columns: 2, successSpans: true };
}

/** Taller on a wide dashboard, same days at every width. */
export function activityChartHeight(width: number): number {
  return width >= DASHBOARD_DESKTOP_MIN ? 180 : 120;
}

export function dashboardGridStyle(layout: DashboardLayout): {
  display: "grid";
  gridTemplateColumns: string;
  overflowX: "clip";
  minWidth: 0;
} {
  return {
    display: "grid",
    gridTemplateColumns: `repeat(${layout.columns}, minmax(0, 1fr))`,
    overflowX: "clip",
    minWidth: 0,
  };
}

export function dashboardPanelStyle(
  layout: DashboardLayout,
  id: DashboardPanelId,
): { gridColumn: string; gridRow?: string; minWidth: 0; overflowX: "clip" } {
  const row = layout.row[id];
  return {
    gridColumn: layout.column[id],
    ...(row ? { gridRow: row } : {}),
    minWidth: 0,
    overflowX: "clip",
  };
}
