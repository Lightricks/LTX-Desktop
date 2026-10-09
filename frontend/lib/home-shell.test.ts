import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { paths } from "../paths.ts";

import { HOME_FEATURES } from "./home-features.ts";
import {
  HOME_SHELL_ENTRIES,
  areHomeFeaturesInNav,
  getHomeShellEntry,
  homeLayoutChrome,
  homeLayoutOutletBlocked,
  homeLayoutPageIds,
  isHomeFeaturePath,
  isHomeShellNavVisible,
  isProjectEditorPath,
  matchHomeShellEntry,
  shellPageHeaderSlot,
} from "./home-shell.ts";

describe("HOME_SHELL_ENTRIES", () => {
  it("keeps generation tools on HOME_FEATURES, not in the shell table", () => {
    const ids = HOME_SHELL_ENTRIES.map((entry) => entry.id);
    assert.deepEqual(ids, [
      "home",
      "assets",
      "dashboard",
      "remote",
      "projects",
      "settings",
      "support",
    ]);
    for (const feature of HOME_FEATURES) {
      assert.equal(ids.includes(feature.id as (typeof ids)[number]), false);
    }
  });

  it("lists Home-layout pages, including Activity Dashboard on desktop and remote", () => {
    assert.deepEqual(homeLayoutPageIds("desktop"), [
      "home",
      "assets",
      "dashboard",
      "remote",
      "projects",
      "settings",
    ]);
    assert.deepEqual(homeLayoutPageIds("remote"), ["home", "assets", "dashboard"]);
  });

  it("keeps Projects under the Home sidebar, with no separate app shell", () => {
    assert.equal(getHomeShellEntry("projects").layout, "home");
  });

  it("labels Projects as GenSpace without moving its route, and keeps Remote in primary nav", () => {
    const projects = getHomeShellEntry("projects");
    assert.equal(projects.label, "Gen Space");
    assert.equal(projects.path, paths.projects);
    assert.equal(getHomeShellEntry("remote").navSlot, "primary");
    assert.equal(getHomeShellEntry("dashboard").label, "Activity Dashboard");
  });

  it("keeps Activity Dashboard in the Support menu, served at /activity-dashboard", () => {
    const dashboard = getHomeShellEntry("dashboard");
    assert.equal(dashboard.navSlot, "support");
    assert.equal(dashboard.path, "/activity-dashboard");
  });
});

describe("matchHomeShellEntry", () => {
  it("matches exact entry paths", () => {
    assert.equal(matchHomeShellEntry(paths.home)?.id, "home");
    assert.equal(matchHomeShellEntry(paths.assets)?.id, "assets");
    assert.equal(matchHomeShellEntry(paths.dashboard)?.id, "dashboard");
    assert.equal(matchHomeShellEntry(paths.projects)?.id, "projects");
    assert.equal(matchHomeShellEntry(paths.settings)?.id, "settings");
    assert.equal(matchHomeShellEntry(paths.remote)?.id, "remote");
  });

  it("lets projects own its /:projectId subtree, nothing else", () => {
    assert.equal(matchHomeShellEntry("/projects/abc")?.id, "projects");
    assert.equal(matchHomeShellEntry("/projects2"), undefined);
    assert.equal(matchHomeShellEntry("/unknown-home-path"), undefined);
  });

  it("never matches feature forms (owned by HOME_FEATURES, not the shell)", () => {
    assert.equal(matchHomeShellEntry(paths.textToVideo), undefined);
  });
});

describe("isProjectEditorPath", () => {
  it("matches only the editor route", () => {
    assert.equal(isProjectEditorPath("/projects/abc"), true);
    assert.equal(isProjectEditorPath(paths.projects), false);
    assert.equal(isProjectEditorPath(paths.home), false);
  });
});

describe("homeLayoutChrome", () => {
  it("shows the header and outlet for plain Home routes", () => {
    assert.deepEqual(homeLayoutChrome(paths.home, true), {
      showHeader: true,
      featureForm: false,
      outletBlocked: false,
    });
  });

  it("hides the header and never blocks the legacy editor", () => {
    assert.deepEqual(homeLayoutChrome("/projects/abc", false), {
      showHeader: false,
      featureForm: true,
      outletBlocked: false,
    });
  });

  it("blocks feature forms when generation is unavailable", () => {
    assert.deepEqual(homeLayoutChrome(paths.textToVideo, false), {
      showHeader: true,
      featureForm: false,
      outletBlocked: true,
    });
  });
});

describe("home shell nav visibility", () => {
  it("keeps Home, Assets, Projects, and Support when Desktop generation is unavailable", () => {
    const visible = HOME_SHELL_ENTRIES.filter((entry) =>
      isHomeShellNavVisible(entry, "desktop", false),
    ).map((entry) => entry.id);
    assert.deepEqual(visible, ["home", "assets", "dashboard", "projects", "settings", "support"]);
    assert.equal(areHomeFeaturesInNav("desktop", false), false);
  });

  it("adds features and Remote when Desktop generation is viable", () => {
    assert.equal(areHomeFeaturesInNav("desktop", true), true);
    assert.equal(
      isHomeShellNavVisible(getHomeShellEntry("remote"), "desktop", true),
      true,
    );
  });

  it("shows Home, features, and Assets on Remote, never Projects or Remote pairing", () => {
    const visible = HOME_SHELL_ENTRIES.filter((entry) =>
      isHomeShellNavVisible(entry, "remote", true),
    ).map((entry) => entry.id);
    assert.deepEqual(visible, ["home", "assets", "dashboard"]);
    assert.equal(areHomeFeaturesInNav("remote", true), true);
  });
});

describe("shellPageHeaderSlot", () => {
  it("keeps the shell header on Home even when generation is unavailable", () => {
    assert.equal(shellPageHeaderSlot(paths.home, true), "shell");
    assert.equal(shellPageHeaderSlot(paths.home, false), "shell");
  });

  it("keeps the shell header on Projects, Settings, and Remote pairing", () => {
    assert.equal(shellPageHeaderSlot(paths.projects, true), "shell");
    assert.equal(shellPageHeaderSlot(paths.settings, false), "shell");
    assert.equal(shellPageHeaderSlot(paths.remote, false), "shell");
  });

  it("puts the header inside Assets and viable workflow pages", () => {
    assert.equal(shellPageHeaderSlot(paths.assets, false), "page");
    assert.equal(shellPageHeaderSlot(paths.textToVideo, true), "page");
  });

  it("uses the shell header when a workflow is blocked, and none on the legacy editor", () => {
    assert.equal(shellPageHeaderSlot(paths.textToVideo, false), "shell");
    assert.equal(shellPageHeaderSlot("/projects/abc", false), null);
  });
});

describe("homeLayoutOutletBlocked", () => {
  it("lets Assets, Projects, and the project editor render when Desktop generation is unavailable", () => {
    assert.equal(homeLayoutOutletBlocked(paths.assets, false), false);
    assert.equal(homeLayoutOutletBlocked(paths.projects, false), false);
    assert.equal(homeLayoutOutletBlocked("/projects/abc", false), false);
    assert.equal(homeLayoutOutletBlocked(paths.home, false), true);
    assert.equal(homeLayoutOutletBlocked(paths.remote, false), true);
    assert.equal(homeLayoutOutletBlocked(paths.textToVideo, false), true);
    assert.equal(homeLayoutOutletBlocked("/unknown-home-path", false), true);
  });

  it("never blocks the Home layout when generation is viable", () => {
    assert.equal(homeLayoutOutletBlocked(paths.home, true), false);
    assert.equal(homeLayoutOutletBlocked(paths.assets, true), false);
    assert.equal(homeLayoutOutletBlocked(paths.textToVideo, true), false);
  });
});

describe("isHomeFeaturePath", () => {
  it("matches feature forms and not Home chrome", () => {
    assert.equal(isHomeFeaturePath(paths.textToVideo), true);
    assert.equal(isHomeFeaturePath(paths.home), false);
    assert.equal(isHomeFeaturePath(paths.assets), false);
    assert.equal(isHomeFeaturePath(paths.remote), false);
  });
});
