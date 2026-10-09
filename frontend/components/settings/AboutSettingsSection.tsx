import { useEffect, useMemo, useState } from "react";

import { Button as DsButton } from "@/ds/Button/Button";
import { Text } from "@/ds/Text/Text";

import type { AppUpdate } from "../../hooks/use-app-update";
import { logger } from "../../lib/logger";
import type { UpdateStatePayload } from "../../../shared/electron-api-schema";

import { SettingToggle } from "./SettingToggle";
import {
  SettingsLinkButton,
  SettingsStack,
  SettingsSubsection,
} from "./SettingsContent";
import contentStyles from "./SettingsContent.module.scss";

function aboutUpdateAction(
  state: UpdateStatePayload,
  onOpenUpdate: () => void,
  onCheckForUpdates: () => void,
  isMac: boolean,
  autoCheckOn: boolean,
): { label: string; onClick?: () => void; disabled?: boolean } {
  if (isMac) {
    switch (state.status) {
      case "checking":
        return { label: "Checking…", disabled: true };
      case "downloading":
        return { label: `Downloading… ${state.percent ?? 0}%`, disabled: true };
      case "downloaded":
        return {
          label: autoCheckOn
            ? "Will install when you quit"
            : "Will still install when you quit",
          disabled: true,
        };
      default:
        return {
          label: "Check for updates",
          onClick: autoCheckOn ? onCheckForUpdates : undefined,
          disabled: !autoCheckOn,
        };
    }
  }
  switch (state.status) {
    case "available":
      return { label: `Update available — v${state.version}`, onClick: onOpenUpdate };
    case "downloaded":
      return { label: "Restart to update", onClick: onOpenUpdate };
    case "checking":
      return { label: "Checking…", disabled: true };
    case "downloading":
      return { label: `Downloading… ${state.percent ?? 0}%`, disabled: true };
    default:
      return { label: "Check for updates", onClick: onCheckForUpdates };
  }
}

function updateStatusDescription(
  state: UpdateStatePayload,
  isMac: boolean,
  autoCheckUpdates: boolean,
): string {
  if (isMac) {
    switch (state.status) {
      case "downloading":
        return `Downloading… ${state.percent ?? 0}%`;
      case "downloaded":
        return autoCheckUpdates
          ? "An update will install when you quit the app."
          : "This update is queued and will install when you quit.";
      case "checking":
        return "Checking for updates…";
      case "not-available":
        return "You're on the latest version.";
      default:
        return autoCheckUpdates
          ? "New versions download in the background and install when you quit."
          : "Turn on automatic updates to install new versions when you quit.";
    }
  }
  switch (state.status) {
    case "available":
      return `Version ${state.version} is available.`;
    case "downloading":
      return `Downloading… ${state.percent ?? 0}%`;
    case "downloaded":
      return "Download complete. Restart to apply the update.";
    case "checking":
      return "Checking for updates…";
    case "not-available":
      return "You're on the latest version.";
    default:
      return "Check manually below, or enable automatic checks.";
  }
}

type AboutDocumentView = "main" | "modelLicense" | "notices";

export function AboutSettingsSection({
  update,
  onOpenUpdate,
  onCheckForUpdates,
}: {
  update: AppUpdate;
  onOpenUpdate: () => void;
  onCheckForUpdates: () => void;
}) {
  const isMac = window.electronAPI.platform === "darwin";
  const [appVersion, setAppVersion] = useState("");
  const [autoCheckUpdates, setAutoCheckUpdates] = useState(true);
  const [view, setView] = useState<AboutDocumentView>("main");
  const [noticesText, setNoticesText] = useState<string | null>(null);
  const [noticesLoading, setNoticesLoading] = useState(false);
  const [modelLicenseText, setModelLicenseText] = useState<string | null>(null);
  const [modelLicenseLoading, setModelLicenseLoading] = useState(false);

  useEffect(() => {
    void window.electronAPI
      .getAppInfo()
      .then((info) => setAppVersion(info.version))
      .catch(() => {});
    void window.electronAPI
      .getAutoCheckUpdates()
      .then((s: { enabled: boolean }) => setAutoCheckUpdates(s.enabled))
      .catch(() => {});
  }, []);

  const updateAction = useMemo(
    () =>
      aboutUpdateAction(
        update.state,
        onOpenUpdate,
        onCheckForUpdates,
        isMac,
        autoCheckUpdates,
      ),
    [autoCheckUpdates, isMac, onCheckForUpdates, onOpenUpdate, update.state],
  );

  const handleToggleAutoCheck = () => {
    const next = !autoCheckUpdates;
    setAutoCheckUpdates(next);
    void window.electronAPI.setAutoCheckUpdates({ enabled: next }).catch(() => {});
  };

  const handleLoadModelLicense = async () => {
    setModelLicenseLoading(true);
    try {
      const text = await window.electronAPI.fetchLicenseText();
      setModelLicenseText(text);
      setView("modelLicense");
    } catch (error) {
      logger.error(`Failed to load model license: ${error}`);
    } finally {
      setModelLicenseLoading(false);
    }
  };

  const handleLoadNotices = async () => {
    setNoticesLoading(true);
    try {
      const text = await window.electronAPI.getNoticesText();
      setNoticesText(text);
      setView("notices");
    } catch (error) {
      logger.error(`Failed to load notices: ${error}`);
    } finally {
      setNoticesLoading(false);
    }
  };

  if (view === "modelLicense" && modelLicenseText) {
    return (
      <LegalDocumentView
        title="LTX-2 model license"
        body={modelLicenseText}
        onBack={() => setView("main")}
      />
    );
  }

  if (view === "notices" && noticesText) {
    return (
      <LegalDocumentView
        title="Third-party notices"
        body={noticesText}
        onBack={() => setView("main")}
      />
    );
  }

  return (
    <SettingsStack loose>
      <div className={contentStyles.aboutAppIntro}>
        <Text as="h3" variant="heading" size="sm">
          LTX Desktop
        </Text>
        <Text as="p" variant="body" size="sm" className={contentStyles.textSecondary}>
          Version {appVersion || "…"}
        </Text>
        <Text as="p" variant="body" size="sm" className={contentStyles.apiKeysDescriptionLine}>
          AI-powered video generation on your machine.
        </Text>
      </div>

      <SettingsSubsection
        showDivider
        title="Updates"
        description={updateStatusDescription(update.state, isMac, autoCheckUpdates)}
        descriptionClassName={contentStyles.apiKeysDescriptionLine}
        descriptionSize="sm"
      >
        <SettingsStack>
          {update.state.message ? (
            <Text as="p" variant="body" size="xs" className="text-fg-danger">
              {update.state.message}
            </Text>
          ) : null}
          <div>
            <DsButton
              appearance="neutral"
              hierarchy="secondary"
              size="md"
              className={contentStyles.settingsControlButton}
              label={updateAction.label}
              disabled={updateAction.disabled}
              onClick={updateAction.onClick}
            />
          </div>
          <SettingToggle
            rowLayout="preference"
            title={isMac ? "Automatic updates" : "Check for updates automatically"}
            description={
              isMac
                ? "Download new versions in the background and install when you quit."
                : "Periodically look for new versions in the background."
            }
            enabled={autoCheckUpdates}
            onToggle={handleToggleAutoCheck}
          />
        </SettingsStack>
      </SettingsSubsection>

      <SettingsSubsection
        showDivider
        title="Software license"
        description="LTX Desktop is licensed under the Apache License, Version 2.0."
        descriptionClassName={contentStyles.apiKeysDescriptionLine}
        descriptionSize="sm"
      />

      <SettingsSubsection
        showDivider
        title="LTX-2 model license"
        description="The LTX-2 model uses the LTX-2 Community License Agreement, accepted during first-run setup."
        descriptionClassName={contentStyles.apiKeysDescriptionLine}
        descriptionSize="sm"
      >
        <SettingsLinkButton onClick={() => void handleLoadModelLicense()}>
          {modelLicenseLoading ? "Loading…" : "View model license"}
        </SettingsLinkButton>
      </SettingsSubsection>

      <SettingsSubsection
        showDivider
        title="Third-party notices"
        description="Open-source components and models used in this app, with their license terms."
        descriptionClassName={contentStyles.apiKeysDescriptionLine}
        descriptionSize="sm"
      >
        <SettingsLinkButton onClick={() => void handleLoadNotices()}>
          {noticesLoading ? "Loading…" : "View third-party notices"}
        </SettingsLinkButton>
      </SettingsSubsection>

      <Text
        as="p"
        variant="body"
        size="xs"
        className={contentStyles.aboutCopyright}
      >
        Copyright © 2026 Lightricks
      </Text>
    </SettingsStack>
  );
}

function LegalDocumentView({
  title,
  body,
  onBack,
}: {
  title: string;
  body: string;
  onBack: () => void;
}) {
  return (
    <SettingsStack loose>
      <div className={contentStyles.aboutDocumentHeader}>
        <Text as="h3" variant="heading" size="xs">
          {title}
        </Text>
        <DsButton
          appearance="neutral"
          hierarchy="plain"
          size="md"
          className={contentStyles.settingsControlButton}
          label="Back"
          onClick={onBack}
        />
      </div>
      <pre className={contentStyles.licensePre}>{body}</pre>
    </SettingsStack>
  );
}
