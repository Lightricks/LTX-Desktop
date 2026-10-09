import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { HOME_SIDEBAR_FEATURED_FEATURES } from "../../lib/home-features.ts";
import { paths } from "../../paths.ts";

import {
  type HomeSidebarActions,
  isHomeSidebarFooterItemActive,
  isHomeSidebarNavItemActive,
  planHomeSidebarFooterItems,
  planHomeSidebarItems,
  planHomeSidebarNavEntries,
  planHomeSidebarNavItems,
  selectHomeSidebarFooterItem,
  selectHomeSidebarNavItem,
} from "./home-sidebar-items.ts";

function recordActions(
  host: "desktop" | "remote",
  calls: Array<{ path: string; clearActiveProject: boolean }>,
): HomeSidebarActions {
  return {
    host,
    goTo: (path, options) => {
      calls.push({ path, clearActiveProject: options.clearActiveProject });
    },
  };
}

describe("planHomeSidebarItems", () => {
  it("pins the Desktop and remote-app sidebars on launch", () => {
    assert.equal(
      planHomeSidebarItems({ host: "desktop", localViable: true }).defaultPinned,
      true,
    );
    assert.equal(
      planHomeSidebarItems({ host: "remote", localViable: true }).defaultPinned,
      true,
    );
  });
});

describe("planHomeSidebarNavItems", () => {
  it("orders pages (Home, Assets, Remote, GenSpace) before Featured tools for a viable Desktop plan", () => {
    const plan = planHomeSidebarItems({ host: "desktop", localViable: true });
    assert.deepEqual(
      planHomeSidebarNavItems(plan).map((item) => ({
        id: item.id,
        path: item.path,
      })),
      [
        { id: "home", path: paths.home },
        { id: "assets", path: paths.assets },
        { id: "remote", path: paths.remote },
        { id: "projects", path: paths.projects },
        ...HOME_SIDEBAR_FEATURED_FEATURES.map((feature) => ({
          id: feature.id,
          path: feature.path,
        })),
      ],
    );
  });

  it("keeps Home, Assets, and features — without Projects or Remote — on the remote-app sidebar", () => {
    const plan = planHomeSidebarItems({ host: "remote", localViable: true });
    const items = planHomeSidebarNavItems(plan);
    assert.deepEqual(
      items.map((item) => item.id),
      ["home", "assets", ...HOME_SIDEBAR_FEATURED_FEATURES.map((feature) => feature.id)],
    );
    for (const item of items) {
      assert.equal(item.path.includes("?"), false);
      assert.equal(item.path.startsWith("/"), true);
    }
  });

  it("omits features and Remote when Desktop generation is not viable", () => {
    const plan = planHomeSidebarItems({ host: "desktop", localViable: false });
    assert.deepEqual(
      planHomeSidebarNavItems(plan).map((item) => item.id),
      ["home", "assets", "projects"],
    );
  });

  it("gates Remote nav visibility on local viability, on Desktop only", () => {
    const viable = planHomeSidebarNavItems(
      planHomeSidebarItems({ host: "desktop", localViable: true }),
    );
    assert.deepEqual(
      viable.find((item) => item.id === "remote"),
      { id: "remote", label: "Remote", path: paths.remote },
    );
    const gated = planHomeSidebarNavItems(
      planHomeSidebarItems({ host: "desktop", localViable: false }),
    );
    assert.equal(
      gated.some((item) => item.id === "remote"),
      false,
    );
    const remoteHost = planHomeSidebarNavItems(
      planHomeSidebarItems({ host: "remote", localViable: true }),
    );
    assert.equal(
      remoteHost.some((item) => item.id === "remote"),
      false,
    );
  });

  it("marks the Featured section after the pages chunk, with every tool flat inside it", () => {
    const plan = planHomeSidebarItems({ host: "desktop", localViable: true });
    const kinds = planHomeSidebarNavEntries(plan).map((entry) =>
      entry.kind === "item" ? entry.item.id : entry.id,
    );
    assert.deepEqual(kinds, [
      "home",
      "assets",
      "remote",
      "projects",
      "featured",
      ...HOME_SIDEBAR_FEATURED_FEATURES.map((feature) => feature.id),
    ]);
    const sections = planHomeSidebarNavEntries(plan).filter(
      (entry) => entry.kind === "section",
    );
    assert.deepEqual(
      sections.map((entry) => (entry.kind === "section" ? entry.label : null)),
      ["Featured"],
    );
  });

  it("lists recent Gen Space projects under the Gen Space nav item", () => {
    const plan = planHomeSidebarItems({ host: "desktop", localViable: true });
    const kinds = planHomeSidebarNavEntries(plan, [], [
      { id: "p1", name: "My timeline" },
    ]).map((entry) =>
      entry.kind === "item"
        ? entry.item.id
        : entry.kind === "recent-project"
          ? `project:${entry.project.id}`
          : entry.id,
    );
    assert.deepEqual(kinds, [
      "home",
      "assets",
      "remote",
      "projects",
      "recent-projects",
      "project:p1",
      "featured",
      "image-to-video",
      "audio-to-video",
      "extend",
    ]);
  });

  it("lists Recent workflows before Featured and omits recent tools from Featured", () => {
    const plan = planHomeSidebarItems({ host: "desktop", localViable: true });
    const kinds = planHomeSidebarNavEntries(plan, [
      "cozy-felt",
      "text-to-video",
    ]).map((entry) => (entry.kind === "item" ? entry.item.id : entry.id));
    assert.deepEqual(kinds, [
      "home",
      "assets",
      "remote",
      "projects",
      "recent",
      "cozy-felt",
      "text-to-video",
      "featured",
      "image-to-video",
      "audio-to-video",
      "extend",
    ]);
  });
});

describe("sidebar Featured list", () => {
  it("lists Image to Video, Audio to Video, and Extend in that order, but not Day to Night or the hidden AlphaGen", () => {
    assert.deepEqual(
      HOME_SIDEBAR_FEATURED_FEATURES.map((feature) => feature.id),
      ["image-to-video", "audio-to-video", "extend"],
    );
  });

  it("moves a recent Day to Night under Recent workflows only", () => {
    const plan = planHomeSidebarItems({ host: "desktop", localViable: true });
    const kinds = planHomeSidebarNavEntries(plan, ["day-to-night"]).map((entry) =>
      entry.kind === "item" ? entry.item.id : entry.id,
    );
    assert.equal(kinds.filter((id) => id === "day-to-night").length, 1);
    assert.ok(kinds.indexOf("day-to-night") < kinds.indexOf("featured"));
  });
});

describe("planHomeSidebarFooterItems", () => {
  it("puts only Settings in the footer on a viable Desktop sidebar", () => {
    const plan = planHomeSidebarItems({ host: "desktop", localViable: true });
    assert.deepEqual(
      planHomeSidebarFooterItems(plan).map((item) => ({
        id: item.id,
        path: item.path,
      })),
      [{ id: "settings", path: paths.settings }],
    );
  });

  it("omits footer items from the remote-app sidebar", () => {
    const plan = planHomeSidebarItems({ host: "remote", localViable: true });
    assert.deepEqual(planHomeSidebarFooterItems(plan), []);
  });

  it("keeps Settings but hides Remote when Desktop generation is not viable", () => {
    const plan = planHomeSidebarItems({ host: "desktop", localViable: false });
    assert.deepEqual(planHomeSidebarFooterItems(plan), [
      { id: "settings", label: "Settings", path: paths.settings },
    ]);
  });
});

describe("selectHomeSidebarNavItem", () => {
  it("clears the active project for Home, Remote, and Projects — not Assets or features", () => {
    const calls: Array<{ path: string; clearActiveProject: boolean }> = [];
    const actions = recordActions("desktop", calls);
    const plan = planHomeSidebarItems({ host: "desktop", localViable: true });
    for (const item of planHomeSidebarNavItems(plan)) {
      selectHomeSidebarNavItem(item, actions);
    }
    assert.deepEqual(calls, [
      { path: paths.home, clearActiveProject: true },
      { path: paths.assets, clearActiveProject: false },
      { path: paths.remote, clearActiveProject: true },
      { path: paths.projects, clearActiveProject: true },
      ...HOME_SIDEBAR_FEATURED_FEATURES.map((feature) => ({
        path: feature.path,
        clearActiveProject: false,
      })),
    ]);
  });

  it("navigates Home, Assets, and features on Remote without project routes", () => {
    const calls: Array<{ path: string; clearActiveProject: boolean }> = [];
    const actions = recordActions("remote", calls);
    const plan = planHomeSidebarItems({ host: "remote", localViable: true });
    for (const item of planHomeSidebarNavItems(plan)) {
      selectHomeSidebarNavItem(item, actions);
    }
    assert.deepEqual(calls, [
      { path: paths.home, clearActiveProject: true },
      { path: paths.assets, clearActiveProject: false },
      ...HOME_SIDEBAR_FEATURED_FEATURES.map((feature) => ({
        path: feature.path,
        clearActiveProject: false,
      })),
    ]);
  });

  it("ignores the desktop-only Remote entry on the remote-app sidebar", () => {
    const calls: Array<{ path: string; clearActiveProject: boolean }> = [];
    selectHomeSidebarNavItem(
      { id: "remote", label: "Remote", path: paths.remote },
      recordActions("remote", calls),
    );
    assert.deepEqual(calls, []);
  });
});

describe("selectHomeSidebarFooterItem", () => {
  it("opens settings in the modal instead of navigating", () => {
    const calls: Array<{ path: string; clearActiveProject: boolean }> = [];
    let settingsOpened = false;
    const actions = {
      ...recordActions("desktop", calls),
      openSettings: () => {
        settingsOpened = true;
      },
    };
    const plan = planHomeSidebarItems({ host: "desktop", localViable: true });
    for (const item of planHomeSidebarFooterItems(plan)) {
      selectHomeSidebarFooterItem(item, actions);
    }
    assert.equal(settingsOpened, true);
    assert.deepEqual(calls, []);
  });
});

describe("isHomeSidebarNavItemActive", () => {
  it("marks Gen Space active only on the project list, not in an open project", () => {
    const projects = { id: "projects" as const, path: paths.projects };
    assert.equal(isHomeSidebarNavItemActive(paths.projects, projects), true);
    assert.equal(isHomeSidebarNavItemActive("/projects/abc", projects), false);
  });

  it("does not prefix-match lookalike paths", () => {
    const projects = { id: "projects" as const, path: paths.projects };
    assert.equal(isHomeSidebarNavItemActive("/projects2", projects), false);
    assert.equal(isHomeSidebarNavItemActive(paths.home, projects), false);
  });

  it("exact-matches non-projects items", () => {
    const home = { id: "home" as const, path: paths.home };
    assert.equal(isHomeSidebarNavItemActive(paths.home, home), true);
    assert.equal(isHomeSidebarNavItemActive(paths.assets, home), false);
    const remote = { id: "remote" as const, path: paths.remote };
    assert.equal(isHomeSidebarNavItemActive(paths.remote, remote), true);
    assert.equal(isHomeSidebarNavItemActive(paths.home, remote), false);
  });
});

describe("isHomeSidebarFooterItemActive", () => {
  it("marks Settings active for its route or the open settings modal", () => {
    const settings = {
      id: "settings" as const,
      label: "Settings",
      path: paths.settings,
    };
    assert.equal(isHomeSidebarFooterItemActive(paths.home, settings), false);
    assert.equal(
      isHomeSidebarFooterItemActive(paths.home, settings, { settingsModalOpen: true }),
      true,
    );
    assert.equal(
      isHomeSidebarFooterItemActive(paths.settings, settings, { settingsModalOpen: false }),
      true,
    );
  });
});
