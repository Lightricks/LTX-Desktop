import { useSyncExternalStore } from "react";

import {
  getResolvedRecentProjects,
  subscribeRecentProjectIds,
  type RecentProjectSidebarItem,
} from "../../lib/home-recent-projects-preference.ts";

/** Live recent Gen Space projects for the sidebar (under Gen Space). */
export function useRecentSidebarProjects(): readonly RecentProjectSidebarItem[] {
  return useSyncExternalStore(
    subscribeRecentProjectIds,
    getResolvedRecentProjects,
    getResolvedRecentProjects,
  );
}
