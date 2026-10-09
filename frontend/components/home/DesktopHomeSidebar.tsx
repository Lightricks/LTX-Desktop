import { useNavigate } from "react-router";
import { useAppChromeControls } from "../AppShell";
import { useAppSettings } from "../../contexts/AppSettingsContext";
import { useProjects } from "../../contexts/ProjectContext";
import { openProject } from "../../lib/project-navigation";
import { HomeSidebar } from "./HomeSidebar";
import {
  planHomeSidebarItems,
  type HomeSidebarActions,
} from "./home-sidebar-items";

export function DesktopHomeSidebar({
  portalContainer,
  openQuickSearch,
}: {
  portalContainer?: HTMLElement | null;
  openQuickSearch?: () => void;
}) {
  const navigate = useNavigate();
  const { openSettings, isSettingsModalOpen } = useAppChromeControls();
  const { activateProject, clearActiveProject, setCurrentTab } = useProjects();
  const { localViable } = useAppSettings();
  const plan = planHomeSidebarItems({ host: "desktop", localViable });
  const actions: HomeSidebarActions = {
    host: "desktop",
    openQuickSearch,
    goTo: (path, { clearActiveProject: shouldClear }) => {
      if (shouldClear) clearActiveProject();
      navigate(path);
    },
    openSettings,
    isSettingsModalOpen,
    openProject: (projectId) => {
      openProject(projectId, { activateProject, setCurrentTab, navigate });
    },
  };
  return (
    <HomeSidebar
      plan={plan}
      actions={actions}
      portalContainer={portalContainer}
    />
  );
}
