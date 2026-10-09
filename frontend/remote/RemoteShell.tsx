import { clsx } from "clsx";
import { Outlet, useLocation } from "react-router";

import layout from "@/components/home/DesktopHomeLayout.module.scss";
import { PageHeader } from "@/components/home/PageHeader";
import { useHistoryShortcuts } from "@/components/home/useHistoryNav";
import {
  isHomeFeaturePath,
  matchHomeShellEntry,
  shellPageHeaderSlot,
} from "@/lib/home-shell";
import { GenerationQueuePanel } from "@/ltx-io/components/GenerationQueue/GenerationQueuePanel";
import { QuickSearchProvider } from "@/ltx-io/components/QuickSearch/QuickSearchContext";
import { RemoteLtxioQuickSearch } from "@/ltx-io/components/QuickSearch/RemoteLtxioQuickSearch";
import { useExploreRuntime } from "@/ltx-io/runtime/ExploreRuntime";

import { RemoteHomeSidebar } from "./RemoteHomeSidebar";
import styles from "./RemoteShell.module.scss";

export function RemoteShell() {
  const { pathname } = useLocation();
  const { api, generationPolling, mediaUrlForAsset, thumbUrlForAsset } =
    useExploreRuntime();
  const shellEntry = matchHomeShellEntry(pathname);
  const isFeatureForm = isHomeFeaturePath(pathname);
  // Remote generation always goes through the paired desktop, so workflow
  // pages get the in-page header the same way a viable desktop does.
  const headerSlot = shellPageHeaderSlot(pathname, true);
  useHistoryShortcuts();

  const queueControl =
    headerSlot == null ? null : (
      <GenerationQueuePanel
        api={api}
        generationPolling={generationPolling}
        mediaUrlForAsset={mediaUrlForAsset}
        thumbUrlForAsset={thumbUrlForAsset}
        placement="header"
      />
    );

  return (
    <QuickSearchProvider>
      <div className={clsx(layout.layout, styles.fillShell)}>
        <RemoteLtxioQuickSearch />
        <RemoteHomeSidebar />
        <div className={layout.main}>
          {headerSlot === "shell" && queueControl != null ? (
            <PageHeader showHistoryNav queueControl={queueControl} />
          ) : null}
          <main
            className={layout.content}
            data-home-scroll-root=""
            data-feature-form={isFeatureForm ? "" : undefined}
            data-home-page={shellEntry?.id === "home" ? "" : undefined}
            data-assets-page={shellEntry?.id === "assets" ? "" : undefined}
          >
            <div className={layout.contentInner}>
              <Outlet
                context={{
                  pageHeaderQueueControl: headerSlot === "page" ? queueControl : null,
                  showHistoryNav: headerSlot != null,
                }}
              />
            </div>
          </main>
        </div>
      </div>
    </QuickSearchProvider>
  );
}
