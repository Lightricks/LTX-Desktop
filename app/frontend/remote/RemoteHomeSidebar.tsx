import { useNavigate } from "react-router";
import { HomeSidebar } from "@/components/home/HomeSidebar";
import { useQuickSearch } from "@/ltx-io/components/QuickSearch/QuickSearchContext";
import {
  planHomeSidebarItems,
  type HomeSidebarActions,
} from "@/components/home/home-sidebar-items";

export function RemoteHomeSidebar() {
  const navigate = useNavigate();
  const { open: openQuickSearch } = useQuickSearch();
  const plan = planHomeSidebarItems({ host: "remote", localViable: true });
  const actions: HomeSidebarActions = {
    host: "remote",
    openQuickSearch,
    goTo: (path) => navigate(path),
  };
  return <HomeSidebar plan={plan} actions={actions} />;
}
