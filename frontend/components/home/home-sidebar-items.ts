import { generatePath } from "react-router";

import {
  getHomeFeature,
  HOME_SIDEBAR_FEATURED_FEATURES,
  isHomeFeatureId,
  type HomeFeatureId,
} from "../../lib/home-features.ts";
import type { RecentProjectSidebarItem } from "../../lib/home-recent-projects-preference.ts";
import { MAX_SIDEBAR_RECENT_ITEMS } from "../../lib/home-sidebar-recents.ts";
import { paths } from "../../paths.ts";
import type { SettingsOpenDetail } from "../../lib/settings-navigation.ts";
import {
  areHomeFeaturesInNav,
  getHomeShellEntry,
  HOME_SHELL_ENTRIES,
  homeShellEntryServesHost,
  isHomeShellNavVisible,
  type HomeShellHost,
} from "../../lib/home-shell.ts";

export type HomeSidebarHost = HomeShellHost;

export type HomeSidebarItemPlan = {
  host: HomeSidebarHost;
  localViable: boolean;
  defaultPinned: boolean;
};

export function planHomeSidebarItems(input: {
  host: HomeSidebarHost;
  localViable: boolean;
}): HomeSidebarItemPlan {
  return {
    host: input.host,
    localViable: input.localViable,
    defaultPinned: true,
  };
}

export type HomeSidebarNavItemId =
  | "home"
  | HomeFeatureId
  | "assets"
  | "remote"
  | "projects";

export type HomeSidebarNavItem = {
  id: HomeSidebarNavItemId;
  label: string;
  path: string;
};

export type HomeSidebarNavEntry =
  | { kind: "item"; item: HomeSidebarNavItem }
  | {
      kind: "recent-project";
      project: RecentProjectSidebarItem;
      path: string;
    }
  | { kind: "section"; id: "recent-projects"; label: "Recent projects" }
  | { kind: "section"; id: "recent"; label: "Recent workflows" }
  | { kind: "section"; id: "featured"; label: "Featured" };

function toFeatureNavItem(feature: {
  id: HomeFeatureId;
  title: string;
  path: string;
}): HomeSidebarNavItem {
  return {
    id: feature.id,
    label: feature.title,
    path: feature.path,
  };
}

export type HomeSidebarFooterItem = {
  id: "settings";
  label: string;
  path: string;
};

export type HomeSidebarActions = {
  host: HomeSidebarHost;
  goTo: (path: string, options: { clearActiveProject: boolean }) => void;
  openQuickSearch?: () => void;
  openProject?: (projectId: string) => void;
  openSettings?: (detail?: SettingsOpenDetail) => void;
  isSettingsModalOpen?: boolean;
};

export function planHomeSidebarNavEntries(
  plan: HomeSidebarItemPlan,
  recentFeatureIds: readonly HomeFeatureId[] = [],
  recentProjects: readonly RecentProjectSidebarItem[] = [],
): HomeSidebarNavEntry[] {
  const entries: HomeSidebarNavEntry[] = [];
  for (const entry of HOME_SHELL_ENTRIES) {
    if (entry.navSlot !== "primary") continue;
    if (!isHomeShellNavVisible(entry, plan.host, plan.localViable)) continue;
    if (entry.path === null) continue;
    entries.push({
      kind: "item",
      item: {
        id: entry.id as Exclude<HomeSidebarNavItemId, HomeFeatureId>,
        label: entry.label,
        path: entry.path,
      },
    });
    if (
      entry.id === "projects" &&
      plan.host === "desktop" &&
      recentProjects.length > 0
    ) {
      entries.push({
        kind: "section",
        id: "recent-projects",
        label: "Recent projects",
      });
      for (const project of recentProjects.slice(0, MAX_SIDEBAR_RECENT_ITEMS)) {
        entries.push({
          kind: "recent-project",
          project,
          path: generatePath(paths.project, { projectId: project.id }),
        });
      }
    }
  }
  if (areHomeFeaturesInNav(plan.host, plan.localViable)) {
    const recentTools = recentFeatureIds.slice(0, MAX_SIDEBAR_RECENT_ITEMS);
    const recentIds = new Set(recentTools);
    if (recentTools.length > 0) {
      entries.push({ kind: "section", id: "recent", label: "Recent workflows" });
      for (const featureId of recentTools) {
        entries.push({
          kind: "item",
          item: toFeatureNavItem(getHomeFeature(featureId)),
        });
      }
    }
    const featured = HOME_SIDEBAR_FEATURED_FEATURES.filter(
      (feature) => !recentIds.has(feature.id),
    );
    if (featured.length > 0) {
      entries.push({ kind: "section", id: "featured", label: "Featured" });
      for (const feature of featured) {
        entries.push({ kind: "item", item: toFeatureNavItem(feature) });
      }
    }
  }
  return entries;
}

export function planHomeSidebarNavItems(
  plan: HomeSidebarItemPlan,
  recentFeatureIds: readonly HomeFeatureId[] = [],
  recentProjects: readonly RecentProjectSidebarItem[] = [],
): HomeSidebarNavItem[] {
  return planHomeSidebarNavEntries(
    plan,
    recentFeatureIds,
    recentProjects,
  ).flatMap((entry) => (entry.kind === "item" ? [entry.item] : []));
}

export function planHomeSidebarFooterItems(
  plan: HomeSidebarItemPlan,
): HomeSidebarFooterItem[] {
  const items: HomeSidebarFooterItem[] = [];
  for (const entry of HOME_SHELL_ENTRIES) {
    if (entry.navSlot !== "footer") continue;
    if (!isHomeShellNavVisible(entry, plan.host, plan.localViable)) continue;
    if (entry.id === "settings") {
      if (entry.path === null) continue;
      items.push({
        id: entry.id,
        label: entry.label,
        path: entry.path,
      });
    }
  }
  return items;
}

export function isHomeSidebarRecentProjectActive(
  pathname: string,
  path: string,
): boolean {
  return pathname === path;
}

export function isHomeSidebarNavItemActive(
  pathname: string,
  item: Pick<HomeSidebarNavItem, "id" | "path">,
): boolean {
  return pathname === item.path;
}

export function isHomeSidebarFooterItemActive(
  pathname: string,
  item: HomeSidebarFooterItem,
  options?: { settingsModalOpen?: boolean },
): boolean {
  return pathname === item.path || options?.settingsModalOpen === true;
}

export function selectHomeSidebarRecentProject(
  projectId: string,
  actions: HomeSidebarActions,
): void {
  if (actions.openProject) {
    actions.openProject(projectId);
    return;
  }
  actions.goTo(generatePath(paths.project, { projectId }), {
    clearActiveProject: false,
  });
}

export function selectHomeSidebarNavItem(
  item: HomeSidebarNavItem,
  actions: HomeSidebarActions,
): void {
  if (isHomeFeatureId(item.id)) {
    actions.goTo(item.path, { clearActiveProject: false });
    return;
  }
  const entry = getHomeShellEntry(item.id);
  if (entry.path === null) return;
  if (!homeShellEntryServesHost(entry, actions.host)) return;
  actions.goTo(entry.path, {
    clearActiveProject: entry.clearsActiveProject,
  });
}

export function selectHomeSidebarFooterItem(
  item: HomeSidebarFooterItem,
  actions: HomeSidebarActions,
): void {
  switch (item.id) {
    case "settings": {
      if (actions.openSettings) {
        actions.openSettings();
        return;
      }
      const entry = getHomeShellEntry(item.id);
      if (entry.path === null) return;
      if (!homeShellEntryServesHost(entry, actions.host)) return;
      actions.goTo(entry.path, {
        clearActiveProject: entry.clearsActiveProject,
      });
      return;
    }
    default: {
      throw new Error(`Unhandled sidebar footer item: ${item.id}`);
    }
  }
}
