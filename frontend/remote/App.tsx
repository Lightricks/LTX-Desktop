import { useCallback, useEffect, useMemo, useState } from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { createBrowserRouter, RouterProvider, useRouteError, useSearchParams } from "react-router";
import { ThemeProvider } from "@/ds/styles/themes/useTheme";
import { queryClient } from "@/lib/queryClient";
import { useDevPanelShortcut } from "@/hooks/use-dev-panel-shortcut";
import { REMOTE_HOME_FEATURE_SCREENS } from "@/lib/home-feature-registry";
import { FeatureChromeLayout } from "@/ltx-io/screens/Feature/chrome/FeatureChromeLayout";
import { ModalProvider } from "@/ltx-io/components/shared/Modal/ModalProvider";
import { loadPackagedExploreSeed } from "@/ltx-io/assets/packaged-explore-assets";
import { DevPanel } from "@/ltx-io/components/DevPanel/DevPanel";
import { ExploreRuntimeProvider, useExploreRuntime } from "@/ltx-io/runtime/ExploreRuntime";
import { AssetsLibraryScreen } from "@/ltx-io/screens/Assets/AssetsLibraryScreen";
import { DashboardScreen } from "@/ltx-io/screens/Dashboard/DashboardScreen";
import { Home } from "@/views/home/Home";
import { api, pairDevice } from "./api";
import { PairingMessage, PairingStatus, type PairingHealth } from "./PairingMessage";
import { createRemoteExploreRuntime } from "./remote-explore-runtime";
import { createRemoteAppRoutes } from "./remote-routes";
import { RemotePairingPage } from "./RemotePairingPage";
import { RemoteShell } from "./RemoteShell";
import styles from "./RemoteShell.module.scss";

function RemoteRouteError() {
  useRouteError();
  return (
    <PairingMessage
      title="Couldn't load this page"
      body="Something went wrong. Keep LTX Desktop open, stay on this Wi-Fi, and return to Home or open the current QR code or link."
    >
      <a className={styles.homeLink} href="/">
        Back to Home
      </a>
    </PairingMessage>
  );
}

// Desktop browsers use the same Ctrl/Cmd+Shift+D as the Electron app; phones have no
// keyboard, so `?devpanel=1` also opens it.
function RemoteDevPanel() {
  const { api } = useExploreRuntime();
  const [searchParams, setSearchParams] = useSearchParams();
  const [open, setOpen] = useState(() => searchParams.get("devpanel") === "1");

  // Closing also drops `?devpanel=1`, so a reload doesn't reopen the panel.
  const close = useCallback(() => {
    setOpen(false);
    if (!searchParams.has("devpanel")) return;
    const next = new URLSearchParams(searchParams);
    next.delete("devpanel");
    setSearchParams(next, { replace: true });
  }, [searchParams, setSearchParams]);

  useDevPanelShortcut(useCallback(() => (open ? close() : setOpen(true)), [open, close]));
  return <DevPanel api={api} open={open} onClose={close} />;
}

function PairedSessionLayout() {
  const runtime = useMemo(
    () =>
      createRemoteExploreRuntime({
        loadPackagedFile: loadPackagedExploreSeed,
      }),
    [],
  );
  const [health, setHealth] = useState<PairingHealth>("checking");

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const paired = await pairDevice();
      if (cancelled) {
        return;
      }
      if (paired !== "ok") {
        setHealth(paired);
        return;
      }
      try {
        const response = await api("/api/session");
        if (!cancelled) {
          setHealth(response.ok ? "ok" : "failed");
        }
      } catch {
        if (!cancelled) {
          setHealth("failed");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (health !== "ok") {
    return <PairingStatus health={health} />;
  }

  return (
    <QueryClientProvider client={queryClient}>
      <ExploreRuntimeProvider value={runtime}>
        <ModalProvider>
          <RemoteShell />
          <RemoteDevPanel />
        </ModalProvider>
      </ExploreRuntimeProvider>
    </QueryClientProvider>
  );
}

const remoteRouter = createBrowserRouter(
  createRemoteAppRoutes({
    errorElement: <RemoteRouteError />,
    pairing: <RemotePairingPage />,
    pairedLayout: <PairedSessionLayout />,
    homeLayoutPages: {
      home: <Home />,
      assets: <AssetsLibraryScreen />,
      dashboard: <DashboardScreen />,
    },
    screens: REMOTE_HOME_FEATURE_SCREENS,
    featureLayout: FeatureChromeLayout,
  }),
);

export function App() {
  return (
    <ThemeProvider defaultPreference="light" className={styles.themeRoot}>
      <RouterProvider router={remoteRouter} />
    </ThemeProvider>
  );
}
