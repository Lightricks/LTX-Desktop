import { createHashRouter } from "react-router";
import { AppShell } from "./components/AppShell";
import { DesktopHomeLayout } from "./components/home/DesktopHomeLayout";
import { createDesktopAppRoutes } from "./desktop-routes";
import { DESKTOP_HOME_FEATURE_ROUTES } from "./lib/home-feature-registry";
import { projectLoader } from "./lib/project-navigation";
import { AssetsLibraryScreen } from "./ltx-io/screens/Assets/AssetsLibraryScreen";
import { DashboardScreen } from "./ltx-io/screens/Dashboard/DashboardScreen";
import { FetcherToolLanding } from "./views/home/FetcherToolLanding";
import { Home } from "./views/home/Home";
import { ProjectsPage } from "./views/Projects";
import { Project } from "./views/Project";
import { RemotePairing } from "./views/RemotePairing";
import { SettingsPage } from "./views/SettingsPage";

const homeFeatureRoutes = DESKTOP_HOME_FEATURE_ROUTES;

const desktopAppRoutes = createDesktopAppRoutes({
  appShell: <AppShell />,
  desktopHomeLayout: <DesktopHomeLayout />,
  homeLayoutPages: {
    home: <Home />,
    assets: <AssetsLibraryScreen />,
    dashboard: <DashboardScreen />,
    remote: <RemotePairing />,
    settings: <SettingsPage />,
    projects: <ProjectsPage />,
  },
  fetcherTool: <FetcherToolLanding />,
  homeFeatureRoutes,
  projectLoader,
  project: <Project />,
});

export const router = createHashRouter(desktopAppRoutes);
