import { useNavigate } from "react-router";

// eslint-disable-next-line no-restricted-imports
import { useAddCustomLoraModal } from "@/ltx-io/screens/Feature/CustomLoraFeatureScreen";

import { useAppSettings } from "../../contexts/AppSettingsContext";
import { useProjects } from "../../contexts/ProjectContext";
import { openProject } from "../../lib/project-navigation";
import { useAppChromeControls } from "../AppShell";

import { HomeSidebar } from "./HomeSidebar";
import { type HomeSidebarActions, planHomeSidebarItems } from "./home-sidebar-items";

export function DesktopHomeSidebar({
  portalContainer,
  openQuickSearch,
}: {
  portalContainer?: HTMLElement | null;
  openQuickSearch?: () => void;
}) {
  const navigate = useNavigate();
  const openAddCustomLora = useAddCustomLoraModal();
  const { openSettings, isSettingsModalOpen } = useAppChromeControls();
  const { activateProject, clearActiveProject, setCurrentTab } = useProjects();
  const { localViable } = useAppSettings();
  const plan = planHomeSidebarItems({ host: "desktop", localViable });
  const actions: HomeSidebarActions = {
    host: "desktop",
    openQuickSearch,
    openAddCustomLora,
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
  return <HomeSidebar plan={plan} actions={actions} portalContainer={portalContainer} />;
}
