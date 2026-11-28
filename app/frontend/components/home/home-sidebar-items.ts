import { generatePath } from "react-router";

import {
  CUSTOM_LORA_HOME_FEATURE,
  type CustomLoraNavId,
  customLoraNavId,
  isCustomLoraNavId,
} from "../../lib/custom-lora.ts";
import {
  HOME_SIDEBAR_FEATURED_FEATURES,
  type HomeFeatureId,
  getHomeFeature,
  isHomeFeatureId,
} from "../../lib/home-features.ts";
import type { RecentProjectSidebarItem } from "../../lib/home-recent-projects-preference.ts";
import {
  HOME_SHELL_ENTRIES,
  type HomeShellHost,
  areHomeFeaturesInNav,
  getHomeShellEntry,
  homeShellEntryServesHost,
  isHomeShellNavVisible,
} from "../../lib/home-shell.ts";
import { MAX_SIDEBAR_RECENT_ITEMS } from "../../lib/home-sidebar-recents.ts";
import type { SettingsOpenDetail } from "../../lib/settings-navigation.ts";
import { customLoraRunPath } from "../../paths.ts";
import { paths } from "../../paths.ts";

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
  | "assistant"
  | HomeFeatureId
  | CustomLoraNavId
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
  /** Desktop only. Opens the add dialog instead of navigating to the add route. */
  openAddCustomLora?: () => void;
  isSettingsModalOpen?: boolean;
};

/** A LoRA the user saved. Only the id and the name matter to the sidebar. */
export type SidebarCustomLora = { id: string; name: string };

/**
 * What the sidebar needs of the custom LoRA feature. `saved` are the saved LoRAs.
 * `recentIds` are the ones the user ran, newest run first.
 */
export type SidebarCustomLoras = {
  saved: readonly SidebarCustomLora[];
  recentIds: readonly string[];
};

const NO_CUSTOM_LORAS: SidebarCustomLoras = { saved: [], recentIds: [] };

function toCustomLoraNavItem(lora: SidebarCustomLora): HomeSidebarNavItem {
  return {
    id: customLoraNavId(lora.id),
    label: lora.name,
    path: customLoraRunPath(lora.id),
  };
}

export function planHomeSidebarNavEntries(
  plan: HomeSidebarItemPlan,
  recentFeatureIds: readonly HomeFeatureId[] = [],
  recentProjects: readonly RecentProjectSidebarItem[] = [],
  customLoras: SidebarCustomLoras = NO_CUSTOM_LORAS,
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
    // Adding a LoRA needs the computer, so a phone does not get the entry.
    if (entry.id === "projects" && plan.host === "desktop" && plan.localViable) {
      entries.push({
        kind: "item",
        item: toFeatureNavItem(getHomeFeature(CUSTOM_LORA_HOME_FEATURE.id)),
      });
    }
    if (entry.id === "projects" && plan.host === "desktop" && recentProjects.length > 0) {
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
    // Every saved LoRA runs as the one Custom LoRA feature. Its place in the recent list goes to
    // the saved LoRAs the user ran, each by its own name.
    const savedById = new Map(customLoras.saved.map((lora) => [lora.id, lora]));
    const recentLoras = customLoras.recentIds.flatMap((id) => {
      const lora = savedById.get(id);
      return lora === undefined ? [] : [lora];
    });
    const recent: HomeSidebarNavItem[] = recentFeatureIds
      .flatMap((featureId): HomeSidebarNavItem[] =>
        featureId === CUSTOM_LORA_HOME_FEATURE.id
          ? recentLoras.map(toCustomLoraNavItem)
          : [toFeatureNavItem(getHomeFeature(featureId))],
      )
      .slice(0, MAX_SIDEBAR_RECENT_ITEMS);
    const recentIds = new Set(recent.map((item) => item.id));
    if (recent.length > 0) {
      entries.push({ kind: "section", id: "recent", label: "Recent workflows" });
      for (const item of recent) entries.push({ kind: "item", item });
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
  customLoras: SidebarCustomLoras = NO_CUSTOM_LORAS,
): HomeSidebarNavItem[] {
  return planHomeSidebarNavEntries(
    plan,
    recentFeatureIds,
    recentProjects,
    customLoras,
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
  if (item.id === CUSTOM_LORA_HOME_FEATURE.id && actions.openAddCustomLora) {
    actions.openAddCustomLora();
    return;
  }
  if (isHomeFeatureId(item.id) || isCustomLoraNavId(item.id)) {
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
