import { useMemo } from "react";
import { Outlet, useLocation } from "react-router";

import { useTheme } from "@/ds/styles/themes/useTheme";
// eslint-disable-next-line no-restricted-imports
import { GenerationQueuePanel } from "@/ltx-io/components/GenerationQueue/GenerationQueuePanel";
// eslint-disable-next-line no-restricted-imports
import { useQuickSearch } from "@/ltx-io/components/QuickSearch/QuickSearchContext";
// eslint-disable-next-line no-restricted-imports
import { QuickSearchProvider } from "@/ltx-io/components/QuickSearch/QuickSearchContext";
// eslint-disable-next-line no-restricted-imports
import { ModalProvider } from "@/ltx-io/components/shared/Modal/ModalProvider";
// eslint-disable-next-line no-restricted-imports
import { ExploreRuntimeProvider } from "@/ltx-io/runtime/ExploreRuntime";
// eslint-disable-next-line no-restricted-imports
import { createDesktopExploreRuntime } from "@/ltx-io/runtime/createDesktopExploreRuntime";

import { useAppSettings } from "../../contexts/AppSettingsContext";
import { GEMINI_KEY_REQUIRED_SETTINGS_DETAIL } from "../../lib/enhance-gemini-key";
import {
  homeLayoutChrome,
  matchHomeShellEntry,
  shellPageHeaderSlot,
} from "../../lib/home-shell";
import {
  LTX_KEY_REQUIRED_SETTINGS_DETAIL,
  TEXT_ENCODING_SETTINGS_DETAIL,
} from "../../lib/open-ltx-api-key-settings";
import { LocalGenerationUnsupportedNotice } from "../LocalGenerationUnsupportedNotice";

import styles from "./DesktopHomeLayout.module.scss";
import { DesktopHomeSidebar } from "./DesktopHomeSidebar";
import { DesktopHomeSupportFab } from "./DesktopHomeSupportFab";
// eslint-disable-next-line no-restricted-imports
import { DesktopLtxioQuickSearch } from "./DesktopLtxioQuickSearch";
import { PageHeader } from "./PageHeader";
import { useHistoryShortcuts } from "./useHistoryNav";

function DesktopHomeLayoutBody({
  rootElement,
}: {
  rootElement: HTMLElement | null | undefined;
}) {
  const { open: openQuickSearch } = useQuickSearch();
  const { localViable, modelsVersion, updateSettings, settings } = useAppSettings();
  const { pathname } = useLocation();
  useHistoryShortcuts();
  const chrome = homeLayoutChrome(pathname, localViable);
  const headerSlot = shellPageHeaderSlot(pathname, localViable);
  const shellEntry = matchHomeShellEntry(pathname);
  const isSettingsPage = shellEntry?.id === "settings";
  const isHomePage = shellEntry?.id === "home";
  const isAssetsPage = shellEntry?.id === "assets";
  const exploreRuntime = useMemo(
    () =>
      createDesktopExploreRuntime(modelsVersion, {
        persistEnhanceProviderPreference: (provider) => {
          updateSettings({ promptEnhancerProviderPreference: provider });
        },
        autoEnhancePrompts: settings.exploreAutoEnhancePrompts,
        onGeminiKeyRequired: () => {
          window.dispatchEvent(
            new CustomEvent("open-settings", {
              detail: GEMINI_KEY_REQUIRED_SETTINGS_DETAIL,
            }),
          );
        },
        usesLtxApiTextEncoding: !settings.useLocalTextEncoder,
        openLtxApiKeySettings: () => {
          window.dispatchEvent(
            new CustomEvent("open-settings", {
              detail: LTX_KEY_REQUIRED_SETTINGS_DETAIL,
            }),
          );
        },
        openTextEncodingSettings: () => {
          window.dispatchEvent(
            new CustomEvent("open-settings", {
              detail: TEXT_ENCODING_SETTINGS_DETAIL,
            }),
          );
        },
      }),
    [
      modelsVersion,
      settings.exploreAutoEnhancePrompts,
      settings.useLocalTextEncoder,
      updateSettings,
    ],
  );

  const queueControl =
    headerSlot == null ? null : (
      <GenerationQueuePanel
        api={exploreRuntime.api}
        generationPolling={exploreRuntime.generationPolling}
        mediaUrlForAsset={exploreRuntime.mediaUrlForAsset}
        thumbUrlForAsset={exploreRuntime.thumbUrlForAsset}
        placement="header"
      />
    );

  return (
    <ExploreRuntimeProvider value={exploreRuntime}>
      <div className={styles.layout}>
        <ModalProvider>
          <DesktopLtxioQuickSearch />
          <DesktopHomeSidebar
            portalContainer={rootElement}
            openQuickSearch={openQuickSearch}
          />
          <div
            className={styles.main}
            data-settings-page={isSettingsPage ? "" : undefined}
          >
            {headerSlot === "shell" && queueControl != null ? (
              <PageHeader showHistoryNav queueControl={queueControl} />
            ) : null}
            <main
              className={styles.content}
              data-home-scroll-root=""
              data-feature-form={chrome.featureForm ? "" : undefined}
              data-settings-page={isSettingsPage ? "" : undefined}
              data-home-page={isHomePage && !chrome.outletBlocked ? "" : undefined}
              data-assets-page={isAssetsPage ? "" : undefined}
            >
              <div className={styles.contentInner}>
                {chrome.outletBlocked ? (
                  <LocalGenerationUnsupportedNotice />
                ) : (
                  <Outlet
                    context={{
                      pageHeaderQueueControl: headerSlot === "page" ? queueControl : null,
                      showHistoryNav: true,
                    }}
                  />
                )}
              </div>
            </main>
          </div>
          <DesktopHomeSupportFab portalContainer={rootElement} />
        </ModalProvider>
      </div>
    </ExploreRuntimeProvider>
  );
}

export function DesktopHomeLayout() {
  const { rootElement } = useTheme();

  return (
    <QuickSearchProvider>
      <DesktopHomeLayoutBody rootElement={rootElement} />
    </QuickSearchProvider>
  );
}
