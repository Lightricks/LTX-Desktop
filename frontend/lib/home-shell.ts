import { matchPath } from "react-router";

import { paths } from "../paths.ts";

import { matchHomeFeature } from "./home-features.ts";

export type HomeShellHost = "desktop" | "remote";
export type HomeShellNavWhen = "always" | "localViable";
export type HomeShellOutletWhen = "always" | "localViable";
export type HomeShellNavSlot = "primary" | "footer" | "floating" | "support";
export type HomeShellLayout = "home" | null;

export type HomeShellEntry = {
  readonly id: "home" | "assets" | "dashboard" | "projects" | "settings" | "remote" | "support";
  readonly label: string;
  readonly path: string | null;
  readonly hosts: readonly HomeShellHost[];
  readonly nav: HomeShellNavWhen;
  readonly outlet: HomeShellOutletWhen;
  readonly navSlot: HomeShellNavSlot;
  readonly layout: HomeShellLayout;
  readonly clearsActiveProject: boolean;
};

/**
 * Chrome around Home — not generation tools.
 *
 * `HOME_FEATURES` owns the feature catalog. `HomeGallery` renders the page.
 * This table owns sidebar presence, the localViable outlet gate, and which
 * routes sit under the Home layout.
 *
 * `nav` vs `outlet` are separate because Home stays in the sidebar when
 * generation is unavailable, but the page itself is the unsupported notice.
 *
 * Table order is sidebar order for the primary slot.
 */
export const HOME_SHELL_ENTRIES = [
  {
    id: "home",
    label: "Home",
    path: paths.home,
    hosts: ["desktop", "remote"],
    nav: "always",
    outlet: "localViable",
    navSlot: "primary",
    layout: "home",
    clearsActiveProject: true,
  },
  {
    id: "assets",
    label: "Assets",
    path: paths.assets,
    hosts: ["desktop", "remote"],
    nav: "always",
    outlet: "always",
    navSlot: "primary",
    layout: "home",
    clearsActiveProject: false,
  },
  {
    id: "dashboard",
    label: "Activity Dashboard",
    path: paths.dashboard,
    hosts: ["desktop", "remote"],
    nav: "always",
    outlet: "always",
    navSlot: "support",
    layout: "home",
    clearsActiveProject: false,
  },
  {
    id: "remote",
    label: "Remote",
    path: paths.remote,
    hosts: ["desktop"],
    nav: "localViable",
    outlet: "localViable",
    navSlot: "primary",
    layout: "home",
    clearsActiveProject: true,
  },
  {
    id: "projects",
    label: "Gen Space",
    path: paths.projects,
    hosts: ["desktop"],
    nav: "always",
    outlet: "always",
    navSlot: "primary",
    layout: "home",
    clearsActiveProject: true,
  },
  {
    id: "settings",
    label: "Settings",
    path: paths.settings,
    hosts: ["desktop"],
    nav: "always",
    outlet: "always",
    navSlot: "footer",
    layout: "home",
    clearsActiveProject: false,
  },
  {
    id: "support",
    label: "Support",
    path: null,
    hosts: ["desktop"],
    nav: "always",
    outlet: "always",
    navSlot: "floating",
    layout: null,
    clearsActiveProject: false,
  },
] as const satisfies readonly HomeShellEntry[];

export type HomeShellEntryId = (typeof HOME_SHELL_ENTRIES)[number]["id"];

type HomeLayoutEntry = Extract<(typeof HOME_SHELL_ENTRIES)[number], { layout: "home" }>;

export type HomeLayoutPageId = HomeLayoutEntry["id"];

export type HomeLayoutRouteSpec = {
  id: HomeLayoutPageId;
  path: string;
};

const HOME_FEATURE_NAV = {
  hosts: ["desktop", "remote"] as const satisfies readonly HomeShellHost[],
  nav: "localViable" as const satisfies HomeShellNavWhen,
};

export function getHomeShellEntry(
  id: HomeShellEntryId,
): (typeof HOME_SHELL_ENTRIES)[number] {
  const entry = HOME_SHELL_ENTRIES.find((candidate) => candidate.id === id);
  if (!entry) {
    throw new Error(`Unknown home shell entry: ${id}`);
  }
  return entry;
}

export function isHomeFeaturePath(pathname: string): boolean {
  return matchHomeFeature(pathname) != null;
}

export function homeShellEntryServesHost(
  entry: Pick<HomeShellEntry, "hosts">,
  host: HomeShellHost,
): boolean {
  return (entry.hosts as readonly HomeShellHost[]).includes(host);
}

export function isHomeShellNavVisible(
  entry: Pick<HomeShellEntry, "hosts" | "nav">,
  host: HomeShellHost,
  localViable: boolean,
): boolean {
  if (!homeShellEntryServesHost(entry, host)) return false;
  return entry.nav === "always" || localViable;
}

export function areHomeFeaturesInNav(host: HomeShellHost, localViable: boolean): boolean {
  return isHomeShellNavVisible(HOME_FEATURE_NAV, host, localViable);
}

export function homeLayoutRouteSpecs(host: HomeShellHost): HomeLayoutRouteSpec[] {
  const specs: HomeLayoutRouteSpec[] = [];
  for (const entry of HOME_SHELL_ENTRIES) {
    if (entry.layout !== "home" || !homeShellEntryServesHost(entry, host)) continue;
    if (entry.path === null) continue;
    specs.push({ id: entry.id, path: entry.path });
  }
  return specs;
}

export function homeLayoutPageIds(host: HomeShellHost): HomeLayoutPageId[] {
  return homeLayoutRouteSpecs(host).map((spec) => spec.id);
}

/**
 * The ONLY place that knows which pathnames belong to which shell entry.
 * `projects` owns its subtree: /projects/:projectId renders inside Home too.
 * Everything else (nav active state, footer, outlet gate, layout chrome)
 * derives nesting from here — never re-implement the prefix rule.
 */
export function matchHomeShellEntry(
  pathname: string,
): (typeof HOME_SHELL_ENTRIES)[number] | undefined {
  return HOME_SHELL_ENTRIES.find((entry) => {
    if (entry.path === null) return false;
    if (entry.id === "projects") {
      return pathname === entry.path || pathname.startsWith(`${entry.path}/`);
    }
    return matchPath({ path: entry.path, end: true }, pathname) != null;
  });
}

/** /projects/:projectId — the legacy editor, rendered outlet-less under Home. */
export function isProjectEditorPath(pathname: string): boolean {
  return matchPath({ path: paths.project, end: true }, pathname) != null;
}

/** True when DesktopHomeLayout should show the unsupported notice. */
export function homeLayoutOutletBlocked(pathname: string, localViable: boolean): boolean {
  if (localViable) return false;
  const entry = matchHomeShellEntry(pathname);
  if (entry === undefined) return true;
  return entry.outlet === "localViable";
}

export type HomeLayoutChrome = {
  /** Header (with Explore queue). Hidden only for the legacy editor. */
  showHeader: boolean;
  /** Feature-form content sizing. */
  featureForm: boolean;
  /** Unsupported notice instead of the outlet. */
  outletBlocked: boolean;
};

/**
 * Where the queue row is mounted.
 * `shell` sits above the scrolling outlet (Home, Projects, Settings, Remote pairing),
 * including when that outlet is the unsupported notice.
 * `page` is inside Assets or a workflow so filters and breadcrumbs share the row.
 */
export type ShellPageHeaderSlot = "shell" | "page";

/**
 * All per-route layout decisions in one place. The legacy-editor exemption
 * lives HERE only: outletBlocked keeps an explicit editor check (robust to
 * future outlet-config flips) instead of relying on the projects prefix rule.
 */
export function homeLayoutChrome(
  pathname: string,
  localViable: boolean,
): HomeLayoutChrome {
  const isLegacyProjectEditor = isProjectEditorPath(pathname);
  return {
    showHeader: !isLegacyProjectEditor,
    featureForm: (isHomeFeaturePath(pathname) && localViable) || isLegacyProjectEditor,
    outletBlocked:
      !isLegacyProjectEditor && homeLayoutOutletBlocked(pathname, localViable),
  };
}

export function shellPageHeaderSlot(
  pathname: string,
  localViable: boolean,
): ShellPageHeaderSlot | null {
  const chrome = homeLayoutChrome(pathname, localViable);
  if (!chrome.showHeader) return null;
  if (matchHomeShellEntry(pathname)?.id === "assets" || chrome.featureForm) {
    return "page";
  }
  return "shell";
}
