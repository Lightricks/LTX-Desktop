import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";

import {
  DASHBOARD_DESKTOP_MIN,
  DASHBOARD_PANELS,
  DASHBOARD_TABLET_MIN,
  activityChartHeight,
  dashboardLayout,
  kpiColumns,
} from "./dashboardLayout.ts";

const WIDE_ORDER = ["kpis", "activity", "loras", "render", "failures", "keep"];

describe("dashboard layout breakpoints", () => {
  it("switches from one column to two exactly at the tablet minimum", () => {
    assert.equal(dashboardLayout(DASHBOARD_TABLET_MIN - 1).columns, 1);
    assert.equal(dashboardLayout(DASHBOARD_TABLET_MIN).columns, 2);
    assert.equal(kpiColumns(DASHBOARD_TABLET_MIN - 1).columns, 2);
    assert.equal(kpiColumns(DASHBOARD_TABLET_MIN).columns, 3);
    assert.equal(kpiColumns(DASHBOARD_TABLET_MIN - 1).successSpans, true);
    assert.equal(kpiColumns(DASHBOARD_TABLET_MIN).successSpans, false);
  });

  it("switches from two columns to three exactly at the desktop minimum", () => {
    assert.equal(dashboardLayout(DASHBOARD_DESKTOP_MIN - 1).columns, 2);
    assert.equal(dashboardLayout(DASHBOARD_DESKTOP_MIN).columns, 3);
    assert.equal(activityChartHeight(DASHBOARD_DESKTOP_MIN - 1), 120);
    assert.equal(activityChartHeight(DASHBOARD_DESKTOP_MIN), 180);
  });

  it("orders panels the same below the desktop minimum and reorders them above it", () => {
    assert.deepEqual(dashboardLayout(DASHBOARD_TABLET_MIN - 1).order, DASHBOARD_PANELS);
    assert.deepEqual(dashboardLayout(DASHBOARD_TABLET_MIN).order, DASHBOARD_PANELS);
    assert.deepEqual(dashboardLayout(DASHBOARD_DESKTOP_MIN).order, WIDE_ORDER);
  });

  it("gives every panel a column at every width", () => {
    for (const width of [
      DASHBOARD_TABLET_MIN - 1,
      DASHBOARD_TABLET_MIN,
      DASHBOARD_DESKTOP_MIN - 1,
      DASHBOARD_DESKTOP_MIN,
    ]) {
      const layout = dashboardLayout(width);
      for (const id of DASHBOARD_PANELS) {
        assert.ok(layout.column[id], `${id} at ${width}px`);
      }
    }
  });

  it("puts the wide layout's charts beside their context", () => {
    const layout = dashboardLayout(DASHBOARD_DESKTOP_MIN);
    assert.equal(layout.column.activity, "1 / span 2");
    assert.equal(layout.column.loras, "3");
    assert.equal(layout.column.render, "1 / span 2");
    assert.equal(layout.row.failures, "3");
    assert.equal(layout.row.keep, "4");
  });

  it("keeps the stylesheet's phone breakpoint one pixel under the tablet minimum", () => {
    const scss = readFileSync(
      new URL("./DashboardScreen.module.scss", import.meta.url),
      "utf8",
    );
    assert.ok(scss.includes(`max-width: ${DASHBOARD_TABLET_MIN - 1}px`));
  });
});
