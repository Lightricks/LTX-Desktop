import {
  Archive,
  Boxes,
  Info,
  KeyRound,
  Settings,
  X,
} from "lucide-react";
import {
  backdropAnimationProps,
  modalAnimationProps,
} from "@ds/lib/popoverAnimationOptions";
import { motion, useReducedMotion } from "framer-motion";
import React, { useEffect, useRef, useState } from "react";

import {
  type AppSettings,
  DEFAULT_GEMINI_MODEL,
  useAppSettings,
} from "../contexts/AppSettingsContext";
import type { AppUpdate } from "../hooks/use-app-update";
import { ApiClient, type ApiSuccessOf } from "../lib/api-client";
import { LtxApiKeyInput } from "./LtxApiKeyInput";
import { ApiKeySettingsBlock } from "./settings/ApiKeySettingsBlock";
import { AboutSettingsSection } from "./settings/AboutSettingsSection";
import { CudaPerformanceSection } from "./settings/CudaPerformanceSection";
import { EnhanceSettingsSection } from "./settings/EnhanceSettingsSection";
import { LegacyTextEncodingSection } from "./settings/LegacyTextEncodingSection";
import { LegacySettingsSection } from "./settings/LegacySettingsSection";
import { ModelsSettingsSection } from "./settings/ModelsSettingsSection";
import pageLayoutStyles from "./settings/SettingsPageLayout.module.scss";
import { SettingsAnchorSection } from "./settings/SettingsAnchorSection";
import { SettingToggle } from "./settings/SettingToggle";
import { ThemePreferenceSetting } from "./settings/ThemePreferenceSetting";
import { settingsSectionElementId } from "../lib/settings-sections";
import { Button as DsButton } from "@/ds/Button/Button";
import { Text } from "@/ds/Text/Text";
import contentStyles from "./settings/SettingsContent.module.scss";
import {
  SettingsCallout,
  SettingsHint,
  SettingsKeyStatus,
  SettingsSelect,
  SettingsStack,
  SettingsSubsection,
} from "./settings/SettingsContent";
import { isSettingsTabAvailable } from "../lib/settings-navigation";
import type {
  SettingsInitialReason,
  SettingsLegacyScrollAnchor,
  SettingsTabId,
} from "../lib/settings-navigation";

export type { SettingsInitialReason, SettingsTabId };

interface SettingsScreenProps {
  isOpen?: boolean;
  onClose?: () => void;
  initialTab?: TabId;
  initialScrollAnchor?: SettingsLegacyScrollAnchor;
  initialReason?: SettingsInitialReason;
  update: AppUpdate;
  onOpenUpdate: () => void;
  onCheckForUpdates: () => void;
}

type TabId = SettingsTabId;

type GeminiModelOption = ApiSuccessOf<"listGeminiModels">["models"][number];

/** Focuses an API Keys tab input once the modal has switched to that tab.
 *  Shared by the LTX and FAL key inputs — each call gets its own ref/pending state.
 *  Pass `sectionRef` + `containerRef` to scroll a whole section (e.g. Gemini heading + banner)
 *  into the tab body; focus then uses preventScroll so the input doesn't yank the banner back
 *  off-screen. */
function useApiKeyFocus(
  isOpen: boolean,
  activeTab: TabId,
  setActiveTab: (tab: TabId) => void,
  sectionRef?: React.RefObject<HTMLElement | null>,
  containerRef?: React.RefObject<HTMLElement | null>,
) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [pending, setPending] = useState(false);

  useEffect(() => {
    if (!pending) return;
    // Closing the modal or leaving API Keys cancels the timeout below; drop `pending` so a
    // later open can't replay scroll/focus that nobody requested this time.
    if (!isOpen || activeTab !== "apiKeys") {
      setPending(false);
      return;
    }

    // Wait for the API Keys tab to paint before scrolling. Clearing `pending` in this
    // effect body used to cancel the scheduled work on the very next render.
    const timeoutId = window.setTimeout(
      () => {
        // Bring the section (and its "Gemini key required" warning) on screen, not just
        // the input — focus alone leaves the banner that explains the prompt off-screen.
        const section = sectionRef?.current;
        const container = containerRef?.current;
        if (section && container) {
          const top =
            container.scrollTop +
            section.getBoundingClientRect().top -
            container.getBoundingClientRect().top;
          container.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
        }
        inputRef.current?.focus({ preventScroll: sectionRef != null });
        setPending(false);
      },
      sectionRef ? 100 : 0,
    );

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [activeTab, pending, isOpen, sectionRef, containerRef]);

  const openAndFocus = () => {
    setActiveTab("apiKeys");
    setPending(true);
  };

  return { inputRef, openAndFocus };
}

function GeminiModelSelect({
  disabled,
  models,
  value,
  onChange,
  inline = false,
}: {
  disabled: boolean;
  models: GeminiModelOption[];
  value: string;
  onChange: (id: string) => void;
  inline?: boolean;
}) {
  const options = models.some((model) => model.id === value)
    ? models
    : [...models, { id: value, displayName: value, description: "" }];
  const selectedDescription =
    options.find((model) => model.id === value)?.description?.trim() ?? "";

  const select = (
    <SettingsSelect
      disabled={disabled}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={(e) => e.stopPropagation()}
      aria-label="Gemini model"
    >
      {options.map((model) => (
        <option key={model.id} title={model.description} value={model.id}>
          {model.displayName}
        </option>
      ))}
    </SettingsSelect>
  );

  if (inline) {
    return select;
  }

  return (
    <SettingsStack>
      {select}
      <SettingsHint>
        Gemini chat models only — powers the Gemini API enhance option and timeline gap
        suggestions. Local Gemma enhance runs on-device and does not use this key.
      </SettingsHint>
      {selectedDescription ? (
        <Text as="p" variant="body" size="xs" className={contentStyles.textSecondary}>
          {selectedDescription}
        </Text>
      ) : null}
    </SettingsStack>
  );
}

export function SettingsScreen({
  isOpen = true,
  onClose,
  initialTab,
  initialScrollAnchor,
  initialReason,
  update,
  onOpenUpdate,
  onCheckForUpdates,
}: SettingsScreenProps) {
  const active = isOpen;
  const {
    settings,
    updateSettings,
    saveLtxApiKey,
    saveFalApiKey,
    saveGeminiApiKey,
    clearLtxApiKey,
    clearFalApiKey,
    clearGeminiApiKey,
    refreshSettings,
    forceApiGenerations,
    cudaAvailable,
  } = useAppSettings();
  const onSettingsChange = (next: AppSettings) => updateSettings(next);
  const [activeTab, setActiveTab] = useState<TabId>("general");
  const tabBodyRef = useRef<HTMLDivElement>(null);
  const geminiSectionRef = useRef<HTMLElement>(null);
  const ltxApiKey = useApiKeyFocus(active, activeTab, setActiveTab);
  const falApiKey = useApiKeyFocus(active, activeTab, setActiveTab);
  const geminiApiKey = useApiKeyFocus(
    active,
    activeTab,
    setActiveTab,
    geminiSectionRef,
    tabBodyRef,
  );
  const [ltxApiKeyInput, setLtxApiKeyInput] = useState("");
  const [falApiKeyInput, setFalApiKeyInput] = useState("");
  const [geminiApiKeyInput, setGeminiApiKeyInput] = useState("");
  type ApiKeyId = "ltx" | "fal" | "gemini";
  // Ref so a second click cannot start before React re-renders the disabled buttons.
  const pendingApiKeyRef = useRef<ApiKeyId | null>(null);
  const [pendingApiKey, setPendingApiKey] = useState<ApiKeyId | null>(null);
  const [ltxKeyError, setLtxKeyError] = useState<string | null>(null);
  const [falKeyError, setFalKeyError] = useState<string | null>(null);
  const [geminiKeyError, setGeminiKeyError] = useState<string | null>(null);
  const showGeminiKeyBanner = initialReason === "geminiKeyRequired";
  const [geminiModelOptions, setGeminiModelOptions] = useState<GeminiModelOption[]>([]);
  const [resolvedGeminiModel, setResolvedGeminiModel] = useState(DEFAULT_GEMINI_MODEL);
  const geminiModelSaveSeq = useRef(0);
  const [analyticsEnabled, setAnalyticsEnabled] = useState(false);

  // Sync active tab with initialTab when the screen becomes active or the prop changes.
  useEffect(() => {
    if (active && initialTab) {
      setActiveTab(initialTab);
    }
  }, [active, initialTab]);

  useEffect(() => {
    if (!active) return;
    if (initialReason === "geminiKeyRequired") {
      setActiveTab("apiKeys");
      geminiApiKey.openAndFocus();
    } else if (initialReason === "ltxKeyRequired") {
      setActiveTab("apiKeys");
      ltxApiKey.openAndFocus();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- openAndFocus is stable for this open
  }, [active, initialReason]);

  const runApiKeyMutation = (
    id: ApiKeyId,
    action: () => Promise<void>,
    onError: (message: string) => void,
    fallback: string,
  ) => {
    if (pendingApiKeyRef.current) return;
    pendingApiKeyRef.current = id;
    setPendingApiKey(id);
    void (async () => {
      try {
        await action();
      } catch (error) {
        onError(error instanceof Error ? error.message : fallback);
      } finally {
        pendingApiKeyRef.current = null;
        setPendingApiKey(null);
      }
    })();
  };

  // Don't let the selection sit on a tab this runtime doesn't show (from initialTab, a deep
  // link, or a stale value). Same availability rule the tab strip uses.
  useEffect(() => {
    if (!isSettingsTabAvailable(activeTab, { forceApiGenerations })) {
      setActiveTab("general");
    }
  }, [forceApiGenerations, activeTab]);

  // Fetch analytics state when the settings surface opens
  useEffect(() => {
    if (!active) return;
    window.electronAPI
      .getAnalyticsState()
      .then((state: { analyticsEnabled: boolean }) =>
        setAnalyticsEnabled(state.analyticsEnabled),
      )
      .catch(() => {});
  }, [active]);

  useEffect(() => {
    const fallbackId = settings.geminiModel.trim() || DEFAULT_GEMINI_MODEL;
    if (!active) return;
    if (!settings.hasGeminiApiKey) {
      setGeminiModelOptions([
        { id: fallbackId, displayName: fallbackId, description: "" },
      ]);
      setResolvedGeminiModel(fallbackId);
      return;
    }

    let cancelled = false;
    const loadGeminiModels = async () => {
      const result = await ApiClient.listGeminiModels();
      if (cancelled) return;
      if (!result.ok) {
        setGeminiModelOptions([
          { id: fallbackId, displayName: fallbackId, description: "" },
        ]);
        setResolvedGeminiModel(fallbackId);
        return;
      }
      setGeminiModelOptions(result.data.models);
      setResolvedGeminiModel(result.data.resolvedModel);
    };
    void loadGeminiModels();
    return () => {
      cancelled = true;
    };
    // Selecting a model calls refreshSettings(), which updates settings.geminiModel. Relisting
    // on that change would hit the backend on every pick and wipe the dropdown if the GET failed.
    // eslint-disable-next-line react-hooks/exhaustive-deps -- fallbackId is only used when the key is missing or the list fails
  }, [active, settings.hasGeminiApiKey]);

  const reduceMotion = useReducedMotion();

  // Analytics handler
  const handleToggleAnalytics = () => {
    const next = !analyticsEnabled;
    setAnalyticsEnabled(next);
    window.electronAPI.setAnalyticsEnabled({ enabled: next }).catch(() => {});
  };

  const showSection = (tabId: TabId) => activeTab === tabId;

  const tabs = [
    { id: "general" as TabId, label: "General", icon: Settings },
    { id: "models" as TabId, label: "Models", icon: Boxes },
    { id: "apiKeys" as TabId, label: "API Keys", icon: KeyRound },
    { id: "legacy" as TabId, label: "Legacy", icon: Archive },
    { id: "about" as TabId, label: "About", icon: Info },
  ].filter((tab) => isSettingsTabAvailable(tab.id, { forceApiGenerations }));

  const panelClassName = pageLayoutStyles.modalPanel;
  const tabBodyClassName = pageLayoutStyles.modalMainPane;

  useEffect(() => {
    if (
      !active ||
      (initialScrollAnchor !== "promptEnhancer" &&
        initialScrollAnchor !== "textEncoding")
    ) {
      return;
    }
    setActiveTab("general");
    // Let General paint before measuring: the row's host div only exists once the tab
    // renders, and a same-tick scroll would land on the previous tab's layout.
    const timeoutId = window.setTimeout(() => {
      const container = tabBodyRef.current;
      const section = container?.querySelector(
        `#${settingsSectionElementId(initialScrollAnchor)}`,
      );
      if (!container || !section) return;
      const top =
        container.scrollTop +
        section.getBoundingClientRect().top -
        container.getBoundingClientRect().top;
      container.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
    }, 100);
    return () => window.clearTimeout(timeoutId);
  }, [active, initialScrollAnchor]);

  const selectSettingsSection = (tabId: TabId) => {
    setActiveTab(tabId);
    if (tabBodyRef.current) {
      tabBodyRef.current.scrollTop = 0;
    }
  };

  useEffect(() => {
    if (tabBodyRef.current) {
      tabBodyRef.current.scrollTop = 0;
    }
  }, [activeTab]);

  const highlightedSection = activeTab;

  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose?.();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isOpen, onClose]);

  const settingsNav = (
    <nav
      className={pageLayoutStyles.nav}
      aria-label="Settings sections"
    >
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const isActive = highlightedSection === tab.id;
        return (
          <DsButton
            key={tab.id}
            hierarchy="plain"
            label={tab.label}
            leftIcon={<Icon />}
            isActive={isActive}
            aria-current={isActive ? "page" : undefined}
            aria-label={tab.label}
            onClick={() => selectSettingsSection(tab.id)}
            className={pageLayoutStyles.navButton}
          />
        );
      })}
    </nav>
  );

  const tabBody = (
    <div ref={tabBodyRef} className={tabBodyClassName}>
          {showSection("general") && (
            <SettingsAnchorSection
              title="General"
            >
            <>
              <SettingsSubsection title="Appearance">
                <ThemePreferenceSetting />
              </SettingsSubsection>

              <SettingsSubsection
                showDivider
                title="Generation"
                description="Prompt rewriting and decode speed for local and API video generation."
                descriptionClassName={contentStyles.apiKeysDescriptionLine}
                descriptionSize="sm"
              >
                <SettingsStack loose>
                  <div id={settingsSectionElementId("promptEnhancer")}>
                    <EnhanceSettingsSection
                      settings={settings}
                      onToggleAutoEnhance={() =>
                        updateSettings({
                          exploreAutoEnhancePrompts: !settings.exploreAutoEnhancePrompts,
                        })
                      }
                      onToggleApiEnhance={() =>
                        updateSettings({
                          promptEnhancerEnabled: !settings.promptEnhancerEnabled,
                        })
                      }
                    />
                  </div>
                  <SettingToggle
                    rowLayout="preference"
                    title="Fast decode"
                    description="Decodes video faster with slightly lower visual fidelity."
                    enabled={settings.useConvVae}
                    onToggle={() =>
                      updateSettings({ useConvVae: !settings.useConvVae })
                    }
                  />
                </SettingsStack>
              </SettingsSubsection>

              {cudaAvailable ? (
                <CudaPerformanceSection
                  settings={settings}
                  onToggleTorchCompile={() =>
                    updateSettings({ useTorchCompile: !settings.useTorchCompile })
                  }
                  onToggleDiffusionStageCache={() =>
                    updateSettings({
                      diffusionStageCacheEnabled: !settings.diffusionStageCacheEnabled,
                    })
                  }
                />
              ) : null}

              <div id={settingsSectionElementId("textEncoding")}>
                <LegacyTextEncodingSection
                  active={active}
                  settings={settings}
                  onOpenLtxApiKeys={() => ltxApiKey.openAndFocus()}
                />
              </div>

              <SettingsSubsection showDivider title="Privacy">
                <SettingToggle
                  rowLayout="preference"
                  title="Anonymous analytics"
                  description="Share anonymous usage data to help improve LTX Desktop. Only basic technical information is collected. Never personal data or generated content."
                  enabled={analyticsEnabled}
                  onToggle={handleToggleAnalytics}
                />
              </SettingsSubsection>
            </>
            </SettingsAnchorSection>
          )}

          {showSection("models") && !forceApiGenerations && (
            <ModelsSettingsSection sectionActive={active} />
          )}

          {showSection("legacy") && (
            <SettingsAnchorSection
              title="Legacy"
            >
            <LegacySettingsSection
              settings={settings}
              onSettingsChange={onSettingsChange}
              onOpenLtxApiKeys={() => ltxApiKey.openAndFocus()}
              onOpenFalApiKeys={() => falApiKey.openAndFocus()}
            />
            </SettingsAnchorSection>
          )}



          {showSection("apiKeys") && (
            <SettingsAnchorSection
              title="API Keys"
            >
            <div className={contentStyles.apiKeysSections}>
              <ApiKeySettingsBlock
                title="LTX API"
                description="Cloud text encoding, prompt enhancement, and API video generation."
                statusBadge={
                  <SettingsKeyStatus
                    configured={settings.hasLtxApiKey}
                    required
                  />
                }
                input={
                  <LtxApiKeyInput
                    ref={ltxApiKey.inputRef}
                    value={ltxApiKeyInput}
                    onChange={(e) => setLtxApiKeyInput(e.target.value)}
                    placeholder={
                      settings.hasLtxApiKey
                        ? "Enter a new key to replace"
                        : "Enter your LTX API key"
                    }
                    stopPropagation
                  />
                }
                replaceDisabled={!ltxApiKeyInput.trim()}
                isSaving={pendingApiKey === "ltx"}
                actionsLocked={pendingApiKey !== null}
                saveError={ltxKeyError}
                keyConfigured={settings.hasLtxApiKey}
                onRemove={() => {
                  runApiKeyMutation(
                    "ltx",
                    async () => {
                      await clearLtxApiKey();
                      setLtxApiKeyInput("");
                    },
                    (message) => window.alert(message),
                    "Could not remove your LTX API key.",
                  );
                }}
                onReplace={() => {
                  const trimmed = ltxApiKeyInput.trim();
                  if (!trimmed) return;
                  setLtxKeyError(null);
                  runApiKeyMutation(
                    "ltx",
                    async () => {
                      await saveLtxApiKey(trimmed);
                      setLtxApiKeyInput("");
                    },
                    setLtxKeyError,
                    "This key isn’t valid.",
                  );
                }}
                getKeyLink={
                  <button
                    type="button"
                    className={contentStyles.apiKeyInlineLink}
                    onClick={(e) => {
                      e.stopPropagation();
                      window.electronAPI.openLtxApiKeyPage();
                    }}
                  >
                    Get API key
                  </button>
                }
              />

              <ApiKeySettingsBlock
                title="Fal"
                description="Generate or edit images with Z Image Turbo when API generations are on."
                statusBadge={
                  settings.hasFalApiKey ? <SettingsKeyStatus configured /> : null
                }
                input={
                  <LtxApiKeyInput
                    ref={falApiKey.inputRef}
                    value={falApiKeyInput}
                    onChange={(e) => setFalApiKeyInput(e.target.value)}
                    placeholder={
                      settings.hasFalApiKey
                        ? "Enter a new key to replace"
                        : "Enter your FAL AI API key"
                    }
                    stopPropagation
                  />
                }
                replaceDisabled={!falApiKeyInput.trim()}
                isSaving={pendingApiKey === "fal"}
                actionsLocked={pendingApiKey !== null}
                saveError={falKeyError}
                keyConfigured={settings.hasFalApiKey}
                onRemove={() => {
                  runApiKeyMutation(
                    "fal",
                    async () => {
                      await clearFalApiKey();
                      setFalApiKeyInput("");
                    },
                    (message) => window.alert(message),
                    "Could not remove your FAL API key.",
                  );
                }}
                onReplace={() => {
                  const trimmed = falApiKeyInput.trim();
                  if (!trimmed) return;
                  setFalKeyError(null);
                  runApiKeyMutation(
                    "fal",
                    async () => {
                      await saveFalApiKey(trimmed);
                      setFalApiKeyInput("");
                    },
                    setFalKeyError,
                    "This key isn’t valid.",
                  );
                }}
                getKeyLink={
                  <button
                    type="button"
                    className={contentStyles.apiKeyInlineLink}
                    onClick={(e) => {
                      e.stopPropagation();
                      window.electronAPI.openFalApiKeyPage();
                    }}
                  >
                    Get FAL API key
                  </button>
                }
              />

              <ApiKeySettingsBlock
                sectionRef={geminiSectionRef}
                className="scroll-mt-2"
                title="Gemini"
                description="Timeline gap suggestions and the Gemini enhance option."
                statusBadge={
                  <SettingsKeyStatus
                    configured={settings.hasGeminiApiKey}
                    missingLabel="Optional"
                  />
                }
                input={
                  <input
                    ref={geminiApiKey.inputRef}
                    type="password"
                    value={geminiApiKeyInput}
                    onChange={(e) => setGeminiApiKeyInput(e.target.value)}
                    placeholder={
                      settings.hasGeminiApiKey
                        ? "Enter a new key to replace"
                        : "Enter your Gemini API key"
                    }
                    onKeyDown={(e) => e.stopPropagation()}
                    className={contentStyles.select}
                  />
                }
                modelSelect={
                  <GeminiModelSelect
                    inline
                    disabled={!settings.hasGeminiApiKey}
                    models={geminiModelOptions}
                    value={resolvedGeminiModel}
                    onChange={(geminiModel) => {
                      const previous = resolvedGeminiModel;
                      const requestId = ++geminiModelSaveSeq.current;
                      setResolvedGeminiModel(geminiModel);
                      void (async () => {
                        const result = await ApiClient.updateSettings({ geminiModel });
                        if (requestId !== geminiModelSaveSeq.current) return;
                        if (!result.ok) {
                          setResolvedGeminiModel(previous);
                          return;
                        }
                        await refreshSettings();
                      })();
                    }}
                  />
                }
                replaceDisabled={!geminiApiKeyInput.trim()}
                isSaving={pendingApiKey === "gemini"}
                actionsLocked={pendingApiKey !== null}
                saveError={geminiKeyError}
                keyConfigured={settings.hasGeminiApiKey}
                onRemove={() => {
                  runApiKeyMutation(
                    "gemini",
                    async () => {
                      await clearGeminiApiKey();
                      setGeminiApiKeyInput("");
                    },
                    (message) => window.alert(message),
                    "Could not remove your Gemini API key.",
                  );
                }}
                onReplace={() => {
                  const trimmed = geminiApiKeyInput.trim();
                  if (!trimmed) return;
                  setGeminiKeyError(null);
                  runApiKeyMutation(
                    "gemini",
                    async () => {
                      await saveGeminiApiKey(trimmed);
                      setGeminiApiKeyInput("");
                    },
                    setGeminiKeyError,
                    "This key isn’t valid.",
                  );
                }}
                getKeyLink={
                  <a
                    href="https://aistudio.google.com/app/apikey"
                    target="_blank"
                    rel="noopener noreferrer"
                    className={contentStyles.apiKeyInlineLink}
                    onClick={(e) => e.stopPropagation()}
                  >
                    Get Gemini API key
                  </a>
                }
                notice={
                  showGeminiKeyBanner ? (
                    <SettingsCallout>
                      <SettingsHint variant="warning">
                        Add a Gemini API key to use the Gemini API enhance option.
                      </SettingsHint>
                    </SettingsCallout>
                  ) : null
                }
              />
            </div>
            </SettingsAnchorSection>
          )}


          {showSection("about") && (
            <SettingsAnchorSection
              title="About"
            >
              <AboutSettingsSection
                update={update}
                onOpenUpdate={onOpenUpdate}
                onCheckForUpdates={onCheckForUpdates}
              />
            </SettingsAnchorSection>
          )}
    </div>

  );

  const settingsFrame = (
    <div className={pageLayoutStyles.modalFrame}>
      <aside className={pageLayoutStyles.modalRail}>
        <Text
          as="h2"
          variant="heading"
          size="lg"
          className={pageLayoutStyles.modalSidebarTitle}
        >
          Settings
        </Text>
        {settingsNav}
      </aside>
      <div className={pageLayoutStyles.modalMain}>
        <div className={pageLayoutStyles.modalContentHeader}>
          <DsButton
            appearance="neutral"
            hierarchy="plain"
            size="sm"
            isIconOnly
            leftIcon={<X className="h-4 w-4" aria-hidden />}
            aria-label="Close settings"
            onClick={onClose}
          />
        </div>
        {tabBody}
      </div>
    </div>
  );

  const instantMotion = reduceMotion
    ? ({ initial: false, transition: { duration: 0 } } as const)
    : null;
  const backdropMotion = instantMotion
    ? { ...instantMotion, animate: { opacity: 1 }, exit: { opacity: 0 } }
    : backdropAnimationProps;
  const panelMotion = instantMotion
    ? {
        ...instantMotion,
        animate: { opacity: 1, scale: 1 },
        exit: { opacity: 0, scale: 1 },
      }
    : modalAnimationProps;

  return (
    <motion.div
      className={pageLayoutStyles.modalOverlay}
      initial={false}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={reduceMotion ? { duration: 0 } : { duration: 0.15 }}
    >
      <motion.div
        className={pageLayoutStyles.modalBackdrop}
        onClick={onClose}
        aria-hidden
        {...backdropMotion}
      />
      <motion.div
        className={panelClassName}
        role="dialog"
        aria-modal
        aria-label="Settings"
        {...panelMotion}
      >
        {settingsFrame}
      </motion.div>
    </motion.div>
  );
}

export type { AppSettings };
