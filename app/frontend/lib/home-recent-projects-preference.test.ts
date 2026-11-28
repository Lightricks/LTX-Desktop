import assert from "node:assert/strict";
import { afterEach, describe, it, mock } from "node:test";

import {
  createDefaultTimeline,
  normalizeProject,
} from "../types/project-model.ts";
import { deleteProjectEntry, writeProject } from "./project-storage.ts";
import { RECENT_SIDEBAR_PROJECT_TTL_MS } from "./home-sidebar-recents.ts";
import {
  clearRecentProjectIds,
  getRecentProjectIds,
  getResolvedRecentProjects,
  MAX_RECENT_PROJECTS,
  recordRecentProjectId,
  removeRecentProjectId,
  subscribeRecentProjectIds,
} from "./home-recent-projects-preference.ts";

function createMemoryStorage(): Storage {
  const store = new Map<string, string>();
  return {
    get length() {
      return store.size;
    },
    clear() {
      store.clear();
    },
    getItem(key: string) {
      return store.get(key) ?? null;
    },
    key(index: number) {
      return [...store.keys()][index] ?? null;
    },
    removeItem(key: string) {
      store.delete(key);
    },
    setItem(key: string, value: string) {
      store.set(key, value);
    },
  };
}

function seedProject(id: string, name: string) {
  const timeline = createDefaultTimeline();
  writeProject(
    id,
    normalizeProject({
      id,
      name,
      createdAt: 1,
      updatedAt: 1,
      assets: [],
      timelines: [timeline],
      activeTimelineId: timeline.id,
    }),
  );
}

describe("home-recent-projects-preference", () => {
  let storage: Storage;

  afterEach(() => {
    clearRecentProjectIds();
    deleteProjectEntry("p1");
    deleteProjectEntry("p2");
    deleteProjectEntry("p3");
    deleteProjectEntry("p4");
    deleteProjectEntry("p5");
    if (storage) {
      Object.defineProperty(globalThis, "localStorage", {
        configurable: true,
        value: storage,
      });
    }
  });

  it("records most-recent-first and dedupes by moving to the front", () => {
    storage = createMemoryStorage();
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: storage,
    });
    seedProject("p1", "One");
    seedProject("p2", "Two");
    recordRecentProjectId("p1");
    recordRecentProjectId("p2");
    recordRecentProjectId("p1");
    assert.deepEqual(getRecentProjectIds(), ["p1", "p2"]);
  });

  it(`keeps at most ${MAX_RECENT_PROJECTS} projects`, () => {
    storage = createMemoryStorage();
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: storage,
    });
    for (const id of ["p1", "p2", "p3", "p4", "p5"]) {
      seedProject(id, id);
    }
    recordRecentProjectId("p1");
    recordRecentProjectId("p2");
    recordRecentProjectId("p3");
    recordRecentProjectId("p4");
    recordRecentProjectId("p5");
    assert.deepEqual(getRecentProjectIds(), ["p5", "p4", "p3", "p2"]);
  });

  it("resolves titles from project storage and drops deleted projects", () => {
    storage = createMemoryStorage();
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: storage,
    });
    seedProject("p1", "Alpha");
    seedProject("p2", "Beta");
    recordRecentProjectId("p1");
    recordRecentProjectId("p2");
    assert.deepEqual(getResolvedRecentProjects(), [
      { id: "p2", name: "Beta" },
      { id: "p1", name: "Alpha" },
    ]);
    deleteProjectEntry("p2");
    assert.deepEqual(getResolvedRecentProjects(), [{ id: "p1", name: "Alpha" }]);
    recordRecentProjectId("p1");
    assert.deepEqual(getRecentProjectIds(), ["p1"]);
  });

  it("drops projects not opened within the 7-day TTL", () => {
    storage = createMemoryStorage();
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: storage,
    });
    seedProject("p1", "Stale");
    seedProject("p2", "Fresh");
    const now = 1_700_000_000_000;
    mock.timers.enable({ now, apis: ["Date"] });
    storage.setItem(
      "ltx-desktop.sidebarRecentProjects",
      JSON.stringify([
        {
          id: "p1",
          lastOpenedAt: now - RECENT_SIDEBAR_PROJECT_TTL_MS - 1,
        },
        { id: "p2", lastOpenedAt: now - 60_000 },
      ]),
    );
    const unsubscribe = subscribeRecentProjectIds(() => {});
    assert.deepEqual(getRecentProjectIds(), ["p2"]);
    unsubscribe();
    mock.timers.reset();
  });

  it("does not write storage when reading the snapshot", () => {
    storage = createMemoryStorage();
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: storage,
    });
    seedProject("p1", "One");
    storage.setItem(
      "ltx-desktop.sidebarRecentProjects",
      JSON.stringify(["p1"]),
    );
    const before = storage.getItem("ltx-desktop.sidebarRecentProjects");
    getRecentProjectIds();
    getResolvedRecentProjects();
    assert.equal(storage.getItem("ltx-desktop.sidebarRecentProjects"), before);
  });

  it("returns a stable snapshot when the list has not changed", () => {
    storage = createMemoryStorage();
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: storage,
    });
    assert.equal(getRecentProjectIds(), getRecentProjectIds());
    seedProject("p1", "One");
    recordRecentProjectId("p1");
    const ids = getRecentProjectIds();
    assert.equal(getRecentProjectIds(), ids);
    const resolved = getResolvedRecentProjects();
    assert.equal(getResolvedRecentProjects(), resolved);
  });

  it("keeps stored recents when recording before a subscriber exists", () => {
    storage = createMemoryStorage();
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: storage,
    });
    seedProject("p1", "One");
    seedProject("p2", "Two");
    const now = Date.now();
    storage.setItem(
      "ltx-desktop.sidebarRecentProjects",
      JSON.stringify([
        { id: "p2", lastOpenedAt: now },
        { id: "p1", lastOpenedAt: now - 1_000 },
      ]),
    );
    recordRecentProjectId("p1");
    assert.deepEqual(getRecentProjectIds(), ["p1", "p2"]);
  });

  it("persists legacy id-only rows so the TTL clock does not reset", () => {
    storage = createMemoryStorage();
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: storage,
    });
    seedProject("p1", "One");
    seedProject("p2", "Two");
    const now = 1_700_000_000_000;
    mock.timers.enable({ now, apis: ["Date"] });
    storage.setItem(
      "ltx-desktop.sidebarRecentProjects",
      JSON.stringify(["p1", "p2"]),
    );
    const unsubscribe = subscribeRecentProjectIds(() => {});
    assert.deepEqual(getRecentProjectIds(), ["p1", "p2"]);
    unsubscribe();
    const stored = JSON.parse(
      storage.getItem("ltx-desktop.sidebarRecentProjects") ?? "[]",
    ) as { id: string; lastOpenedAt: number }[];
    assert.deepEqual(
      stored.map((record) => record.lastOpenedAt),
      [now, now],
    );
    mock.timers.tick(5_000);
    assert.deepEqual(getRecentProjectIds(), ["p1", "p2"]);
    const again = JSON.parse(
      storage.getItem("ltx-desktop.sidebarRecentProjects") ?? "[]",
    ) as { id: string; lastOpenedAt: number }[];
    assert.deepEqual(
      again.map((record) => record.lastOpenedAt),
      [now, now],
    );
    mock.timers.reset();
  });

  it("notifies subscribers when the list changes", () => {
    storage = createMemoryStorage();
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: storage,
    });
    seedProject("p1", "One");
    let calls = 0;
    const unsubscribe = subscribeRecentProjectIds(() => {
      calls += 1;
    });
    recordRecentProjectId("p1");
    assert.equal(calls, 1);
    unsubscribe();
    removeRecentProjectId("p1");
    assert.equal(calls, 1);
  });
});
