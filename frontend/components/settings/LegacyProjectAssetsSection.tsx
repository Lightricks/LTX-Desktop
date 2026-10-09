import { useEffect, useState } from "react";

import { Button as DsButton } from "@/ds/Button/Button";

import {
  SettingsFieldRow,
  SettingsReadonlyField,
  SettingsSubsection,
} from "./SettingsContent";
import contentStyles from "./SettingsContent.module.scss";

export function LegacyProjectAssetsSection() {
  const [projectAssetsPath, setProjectAssetsPath] = useState("");

  useEffect(() => {
    void window.electronAPI
      .getProjectAssetsPath()
      .then((path: string) => setProjectAssetsPath(path))
      .catch(() => {});
  }, []);

  return (
    <SettingsSubsection
      title="Project assets"
      description="Where generated videos and images are stored. Each project gets its own subfolder."
      descriptionClassName={contentStyles.apiKeysDescriptionLine}
      descriptionSize="sm"
    >
      <SettingsFieldRow>
        <SettingsReadonlyField value={projectAssetsPath} />
        <DsButton
          appearance="neutral"
          hierarchy="secondary"
          size="md"
          className={contentStyles.settingsControlButton}
          label="Change…"
          onClick={async () => {
            const result =
              await window.electronAPI.openProjectAssetsPathChangeDialog();
            if (result.success) {
              setProjectAssetsPath(result.path);
            }
          }}
        />
      </SettingsFieldRow>
    </SettingsSubsection>
  );
}
