import { Button } from "@ds/Button/Button";
import { Tooltip } from "@ds/Tooltip/Tooltip";
import { MessageSquare } from "lucide-react";
import { useMemo, useState } from "react";
import { Outlet, useLocation, useNavigate } from "react-router";

import { useTheme } from "@/ds/styles/themes/useTheme";
// eslint-disable-next-line no-restricted-imports
import { DesktopAssistantPanel } from "@/ltx-io/components/Assistant/DesktopAssistantPanel";
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
import { paths } from "@/paths";

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
import { HistoryNavButtons } from "./HistoryNavButtons";
import { PageHeader } from "./PageHeader";
import { CautiousModePill } from "./CautiousModePill";
import { useHistoryShortcuts } from "./useHistoryNav";

function DesktopHomeLayoutBody({
  rootElement,
}: {
  rootElement: HTMLElement | null | undefined;
}) {
  const { open: openQuickSearch } = useQuickSearch();
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [assistantAttachment, setAssistantAttachment] = useState<{ id: string } | null>(
    null,
  );
  const navigate = useNavigate();
  const {
    localViable,
    modelsVersion,
    updateSettings,
    settings,
    selectAssistant,
    assistantSelectionPending,
    budgetModeSaving,
    neverSplitSaving,
  } = useAppSettings();
  const { pathname } = useLocation();
  useHistoryShortcuts();
  const chrome = homeLayoutChrome(pathname, localViable);
  const headerSlot = shellPageHeaderSlot(pathname, localViable);
  const shellEntry = matchHomeShellEntry(pathname);
  const isSettingsPage = shellEntry?.id === "settings";
  const isHomePage = shellEntry?.id === "home";
  const isAssetsPage = shellEntry?.id === "assets";
  const isAssistantPage = shellEntry?.id === "assistant";
  const assistantAvailable = !isSettingsPage && shellEntry?.id !== "projects";
  const nagUnavailableReason =
    settings.nagScale <= 0
      ? "Negative prompt guidance (NAG) is turned off in Settings (strength is 0)."
      : settings.hasLtxApiKey && !settings.useLocalTextEncoder
        ? "Negative prompt guidance (NAG) requires the local text encoder and is skipped while LTX API text encoding is active."
        : null;
  const exploreRuntime = useMemo(
    () => ({
      ...createDesktopExploreRuntime(modelsVersion, {
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
        nagUnavailableReason,
        showNagStrengthControls: settings.showNagStrengthControls,
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
        budgetMode: settings.budgetMode,
        budgetModeSaving,
        advancedControls: settings.advancedControls,
        neverSplitIntoParts: settings.neverSplitIntoParts,
        neverSplitSaving,
        budgetRiskConfirm: {
          skip: settings.skipBudgetRiskConfirm,
          setSkip: (skip) => updateSettings({ skipBudgetRiskConfirm: skip }),
        },
      }),
      askAssistant: (id: string) => {
        setAssistantAttachment({ id });
        setAssistantOpen(true);
      },
      assistantSettings: {
        provider: settings.assistantProvider,
        localModel: settings.assistantLocalModel,
        geminiModel: settings.geminiModel,
        hasGeminiApiKey: settings.hasGeminiApiKey,
        select: selectAssistant,
        switching: assistantSelectionPending,
        openSettings: (fix?: "geminiKey" | "localModel") =>
          window.dispatchEvent(
            new CustomEvent("open-settings", {
              detail:
                fix === "geminiKey"
                  ? GEMINI_KEY_REQUIRED_SETTINGS_DETAIL
                  : fix === "localModel"
                    ? { tab: "models" }
                    : { tab: "general", scrollAnchor: "assistant" },
            }),
          ),
      },
    }),
    [
      modelsVersion,
      settings.budgetMode,
      budgetModeSaving,
      settings.advancedControls,
      settings.neverSplitIntoParts,
      neverSplitSaving,
      settings.skipBudgetRiskConfirm,
      settings.exploreAutoEnhancePrompts,
      settings.useLocalTextEncoder,
      nagUnavailableReason,
      settings.showNagStrengthControls,
      settings.assistantProvider,
      settings.assistantLocalModel,
      settings.geminiModel,
      settings.hasGeminiApiKey,
      updateSettings,
      selectAssistant,
      assistantSelectionPending,
    ],
  );

  const queueControl =
    headerSlot == null ? null : (
      <div className={styles.headerActions}>
        {assistantAvailable && !isAssistantPage && (
          <Tooltip content="Video assistant">
            <Button
              appearance="neutral"
              hierarchy="plain"
              size="md"
              isIconOnly
              isActive={assistantOpen}
              leftIcon={<MessageSquare size={20} />}
              aria-label="Video assistant"
              aria-expanded={assistantOpen}
              onClick={() => setAssistantOpen((value) => !value)}
            />
          </Tooltip>
        )}
        <GenerationQueuePanel
          api={exploreRuntime.api}
          generationPolling={exploreRuntime.generationPolling}
          mediaUrlForAsset={exploreRuntime.mediaUrlForAsset}
          thumbUrlForAsset={exploreRuntime.thumbUrlForAsset}
          placement="header"
        />
      </div>
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
            hidden={isAssistantPage}
            data-settings-page={isSettingsPage ? "" : undefined}
            data-assistant-open={assistantOpen && assistantAvailable ? "" : undefined}
          >
            {headerSlot === "shell" && queueControl != null && !isAssistantPage ? (
              <PageHeader
                showHistoryNav
                queueControl={queueControl}
                beforeActions={<CautiousModePill />}
              />
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
                      beforeHeaderActions: <CautiousModePill />,
                    }}
                  />
                )}
              </div>
            </main>
          </div>
          <DesktopAssistantPanel
            open={isAssistantPage || (assistantOpen && assistantAvailable)}
            fullPage={isAssistantPage}
            headerLeading={<HistoryNavButtons />}
            attachmentRequest={assistantAttachment}
            headerActions={
              isAssistantPage ? (
                <>
                  <CautiousModePill />
                  {queueControl}
                </>
              ) : (
                queueControl
              )
            }
            onExpand={() => navigate(paths.assistant)}
            onDock={() => {
              setAssistantOpen(true);
              navigate(paths.assets);
            }}
            onClose={() => setAssistantOpen(false)}
          />
          <DesktopHomeSupportFab
            portalContainer={rootElement}
            besidePanel={!isAssistantPage && assistantOpen && assistantAvailable}
          />
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
